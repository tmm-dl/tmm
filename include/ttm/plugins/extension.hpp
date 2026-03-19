/**
 * @file extension.hpp
 * @brief C++ virtual interfaces for TTM plugin extension points.
 *
 * @details
 * These interfaces are what host-side code works with at runtime.  Plugins
 * that are loaded as WASM modules are wrapped by adapters (CSourceAdapter)
 * that delegate to the C vtables declared in abi.h.  Native plugins may
 * subclass these interfaces directly.
 *
 * @see ttm::plugins::PluginManager  Maintains the registries of these objects
 * @see abi.h                        Underlying C ABI that WASM plugins implement
 */

#ifndef TTM_PLUGINS_EXTENSION_HPP
#define TTM_PLUGINS_EXTENSION_HPP

#include <cstddef>
#include <cstdint>
#include <ios>
#include <istream>
#include <memory>
#include <streambuf>
#include <string>
#include <string_view>
#include <vector>

namespace ttm::plugins {

	/* =========================================================================
 * @defgroup ext_byte_reader IByteReader — sequential byte stream
 * @{
 * ====================================================================== */

	/**
 * @brief Abstract sequential (optionally seekable) byte-stream reader.
 *
 * @details
 * Returned by IDatasetSource::open().  Callers can drive reads directly via
 * read() / seek(), or wrap the reader in a standard `std::istream` using
 * as_stream() for compatibility with libraries that expect C++ stream semantics.
 *
 * @par Example — direct reads
 * @code{.cpp}
 * auto reader = source.open("file:///data/train.arrow");
 * if (!reader) throw std::runtime_error("failed to open dataset");
 *
 * std::array<std::byte, 4096> buf;
 * while (true) {
 *     auto n = reader->read(buf.data(), buf.size());
 *     if (n == 0) break;   // EOF
 *     if (n < 0)  throw std::runtime_error("read error");
 *     process(buf.data(), static_cast<std::size_t>(n));
 * }
 * @endcode
 *
 * @par Example — via std::istream
 * @code{.cpp}
 * auto reader = source.open("gh:owner/repo/data.arrow");
 * std::istream& stream = reader->as_stream();
 * arrow::io::ReadableFile::Open(stream);  // hypothetical Arrow API
 * @endcode
 *
 * @see IDatasetSource::open  Returns a unique_ptr<IByteReader>
 * @see as_stream             Lazily adapts this reader to std::istream
 */
	class IByteReader {
	public:
		IByteReader() = default;
		virtual ~IByteReader();

		IByteReader(const IByteReader&) = delete;
		IByteReader& operator=(const IByteReader&) = delete;
		IByteReader(IByteReader&&) = delete;
		IByteReader& operator=(IByteReader&&) = delete;

		/**
     * @brief Read up to `n` bytes from the stream.
     *
     * @param[out] buf  Destination buffer; must have capacity of at least `n` bytes.
     * @param[in]  n    Maximum number of bytes to read.
     * @return Number of bytes actually written to `buf` (>0),
     *         0 at end-of-stream, or a negative value on error.
     */
		virtual std::streamsize read(std::byte* buf, std::streamsize n) = 0;

		/**
     * @brief Query whether this reader supports random access.
     * @return `true` if seek() is supported; `false` otherwise.
     * @see seek
     */
		[[nodiscard]] virtual bool seekable() const noexcept = 0;

		/**
     * @brief Seek to a position within the stream.
     *
     * @param[in] off  Byte offset relative to `dir`.
     * @param[in] dir  Origin: `std::ios_base::beg`, `cur`, or `end`.
     * @return New absolute byte offset from the beginning of the stream,
     *         or -1 if the reader is not seekable or an error occurred.
     * @see seekable
     */
		virtual std::streampos seek(std::streamoff off, std::ios_base::seekdir dir) = 0;

		/**
     * @brief Adapt this reader to a `std::istream`.
     *
     * @details
     * Creates a `std::streambuf` subclass that delegates to read() and seek(),
     * then wraps it in a `std::istream`.  Both are owned by this IByteReader
     * and share its lifetime.  The stream is created lazily on the first call;
     * subsequent calls return the same object.
     *
     * @return Reference to the adaptor `std::istream`.  Valid for the lifetime
     *         of this IByteReader.
     *
     * @par Example
     * @code{.cpp}
     * std::istream& in = reader->as_stream();
     * json data = json::parse(in);
     * @endcode
     *
     * @see detail::ByteReaderBuf  Internal streambuf implementation
     */
		std::istream& as_stream();

	private:
		std::unique_ptr<std::streambuf> streambuf;
		std::unique_ptr<std::istream> stream;
	};

	/** @} */

	/* =========================================================================
 * @defgroup ext_sources IDatasetSource — URI-based dataset factory
 * @{
 * ====================================================================== */

	/**
 * @brief Factory that opens dataset URIs and returns byte-stream readers.
 *
 * @details
 * A single source implementation may handle multiple URI schemes.  The
 * PluginManager dispatches open() calls based on the scheme prefix (everything
 * up to and including the first `:`).
 *
 * @par Example — implementing a custom source
 * @code{.cpp}
 * class S3Source : public ttm::plugins::IDatasetSource {
 * public:
 *     std::vector<std::string> schemes() const override {
 *         return {"s3:", "s3a:"};
 *     }
 *     std::unique_ptr<IByteReader> open(std::string_view uri) override {
 *         return std::make_unique<S3Reader>(uri);
 *     }
 * };
 * @endcode
 *
 * @par Example — using a source through the PluginManager
 * @code{.cpp}
 * auto* src = manager.find_source("s3:");
 * if (!src) throw std::runtime_error("no handler for s3: URIs");
 * auto reader = src->open("s3://my-bucket/train.arrow");
 * @endcode
 *
 * @see IByteReader                   Returned by open()
 * @see PluginManager::find_source    Look up a source by scheme
 */
	class IDatasetSource {
	public:
		IDatasetSource() = default;
		virtual ~IDatasetSource() = default;

		IDatasetSource(const IDatasetSource&) = delete;
		IDatasetSource& operator=(const IDatasetSource&) = delete;
		IDatasetSource(IDatasetSource&&) = delete;
		IDatasetSource& operator=(IDatasetSource&&) = delete;

		/**
     * @brief URI schemes handled by this source.
     *
     * @details Each entry is a scheme prefix including the trailing colon,
     *          e.g. `"file:"`, `"gh:"`, `"hf:"`.
     *
     * @return Non-empty list of scheme strings owned by the implementation.
     */
		[[nodiscard]] virtual std::vector<std::string> schemes() const = 0;

		/**
     * @brief Open a URI and return a reader for its contents.
     *
     * @param[in] uri  Full URI, including the scheme prefix.
     * @return Unique pointer to a reader, or `nullptr` on failure.  Failures
     *         should be logged via the host API before returning nullptr.
     *
     * @see IByteReader
     */
		virtual std::unique_ptr<IByteReader> open(std::string_view uri) = 0;
	};

	/** @} */

	/* =========================================================================
 * @defgroup ext_transforms ITransform — record-batch transformation
 * @{
 * ====================================================================== */

	/**
 * @brief Transforms an Arrow RecordBatch (e.g. tokenisation, normalisation).
 *
 * @details
 * Full interface is TBD when the training pipeline design is finalised.
 * Plugins may register transforms via #ttm_host_api::register_transform.
 *
 * @see ttm_transform_vtable  Underlying C vtable
 */
	class ITransform {
	public:
		ITransform() = default;
		virtual ~ITransform() = default;

		ITransform(const ITransform&) = delete;
		ITransform& operator=(const ITransform&) = delete;
		ITransform(ITransform&&) = delete;
		ITransform& operator=(ITransform&&) = delete;

		/* Interface to be defined. */
	};

	/** @} */

	/* =========================================================================
 * @defgroup ext_tasks ITask — ML task type
 * @{
 * ====================================================================== */

	/**
 * @brief Defines an ML task type (e.g. text-classification, token-classification).
 *
 * @details
 * Task types correspond to the "tasks" field in HuggingFace Dataset Cards.
 * The same task may be registered under multiple aliases (e.g.
 * `"ner"` and `"token-classification"`).  Full interface is TBD.
 *
 * @see ttm_task_vtable  Underlying C vtable
 */
	class ITask {
	public:
		ITask() = default;
		virtual ~ITask() = default;

		ITask(const ITask&) = delete;
		ITask& operator=(const ITask&) = delete;
		ITask(ITask&&) = delete;
		ITask& operator=(ITask&&) = delete;

		/** Canonical task name, e.g. "text-classification". */
		[[nodiscard]] virtual std::string_view name() const = 0;

		/** All registered aliases for this task (may be empty). */
		[[nodiscard]] virtual std::vector<std::string_view> aliases() const = 0;

		/** Input feature names expected from the dataset schema. */
		[[nodiscard]] virtual std::vector<std::string_view> input_features() const = 0;

		/** Label feature name expected from the dataset schema. */
		[[nodiscard]] virtual std::string_view label_feature() const = 0;

		/** Default evaluation metric names. */
		[[nodiscard]] virtual std::vector<std::string_view> default_metrics() const = 0;
	};

	/** @} */

	/* =========================================================================
 * @defgroup ext_metrics IMetric — evaluation metric
 * @{
 * ====================================================================== */

	/**
 * @brief Computes an evaluation metric (e.g. F1, accuracy, BLEU).
 *
 * @details
 * Metrics may be registered under multiple aliases.  Full interface is TBD.
 *
 * @see ttm_metric_vtable  Underlying C vtable
 */
	class IMetric {
	public:
		IMetric() = default;
		virtual ~IMetric() = default;

		IMetric(const IMetric&) = delete;
		IMetric& operator=(const IMetric&) = delete;
		IMetric(IMetric&&) = delete;
		IMetric& operator=(IMetric&&) = delete;

		/* Interface to be defined. */
	};

	/** @} */

} // namespace ttm::plugins

#endif /* TTM_PLUGINS_EXTENSION_HPP */
