/**
 * @file byte_reader.cpp
 * @brief IByteReader::as_stream() implementation via ByteReaderBuf.
 *
 * @details
 * ByteReaderBuf is a std::streambuf subclass that delegates all I/O to an
 * IByteReader.  It is lazily created by IByteReader::as_stream() and owned
 * by the IByteReader instance.
 *
 * Buffer design:
 * - A fixed-size read buffer (get area) is used so that the streambuf does
 *   not call IByteReader::read() for every single byte.
 * - The put area is left empty (this is a read-only buffer).
 * - seekoff() is forwarded to IByteReader::seek() only when seekable().
 */

#include <ttm/plugins/extension.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <ios>
#include <istream>
#include <memory>
#include <streambuf>

namespace ttm::plugins {

/* =========================================================================
 * detail::ByteReaderBuf — internal std::streambuf adapter
 * ====================================================================== */

namespace detail {

/**
 * @brief std::streambuf that reads from an IByteReader.
 *
 * @details
 * Instances are created and owned by IByteReader::as_stream().  The reader
 * must outlive this buffer.
 *
 * @see IByteReader::as_stream
 */
class ByteReaderBuf final : public std::streambuf {
public:
    /**
     * @param[in] reader  The reader this buffer delegates to.
     *                    Must remain valid for the lifetime of this object.
     */
    explicit ByteReaderBuf(IByteReader& reader)
        : reader_(reader)
    {
        /* Start with an empty get area; underflow() will fill it. */
        setg(buf_.data(), buf_.data(), buf_.data());
    }

protected:
    /* ------------------------------------------------------------------
     * Read interface
     * --------------------------------------------------------------- */

    /**
     * @brief Refill the internal buffer from the underlying reader.
     * @details Called by the base class whenever the get area is exhausted.
     * @return The next character as an unsigned char cast to int_type,
     *         or traits_type::eof() at end of stream.
     */
    int_type underflow() override
    {
        const auto n = reader_.read(
            reinterpret_cast<std::byte*>(buf_.data()),
            static_cast<std::streamsize>(buf_.size()));

        if (n <= 0) {
            return traits_type::eof();
        }

        setg(buf_.data(), buf_.data(), buf_.data() + n);
        return traits_type::to_int_type(*gptr());
    }

    /* ------------------------------------------------------------------
     * Seek interface
     * --------------------------------------------------------------- */

    /**
     * @brief Forward seek requests to the underlying IByteReader.
     * @details Seeking is only supported when IByteReader::seekable() is true.
     *
     * @param[in] off   Byte offset relative to `dir`.
     * @param[in] dir   Origin direction.
     * @param[in] which Must include std::ios_base::in; out is rejected.
     * @return New stream position, or pos_type(off_type(-1)) on failure.
     */
    pos_type seekoff(off_type off,
                     std::ios_base::seekdir dir,
                     std::ios_base::openmode which) override
    {
        /* This buffer is read-only */
        if (!(which & std::ios_base::in)) {
            return pos_type(off_type(-1));
        }

        if (!reader_.seekable()) {
            return pos_type(off_type(-1));
        }

        const auto pos = reader_.seek(off, dir);
        if (pos == std::streampos(-1)) {
            return pos_type(off_type(-1));
        }

        /* After seeking, the get area is stale — reset it to empty so that
         * the next read triggers underflow() to refill from the new position. */
        setg(buf_.data(), buf_.data(), buf_.data());
        return pos_type(pos);
    }

    /**
     * @brief Absolute seek — delegates to seekoff(off, beg, which).
     */
    pos_type seekpos(pos_type sp,
                     std::ios_base::openmode which) override
    {
        return seekoff(off_type(sp), std::ios_base::beg, which);
    }

private:
    IByteReader& reader_;

    /** Internal read buffer — 64 KiB to amortise plugin boundary crossings. */
    std::array<char, 65536> buf_;
};

} // namespace detail

/* =========================================================================
 * IByteReader implementation
 * ====================================================================== */

IByteReader::~IByteReader() = default;

std::istream& IByteReader::as_stream()
{
    if (!streambuf_) {
        streambuf_ = std::make_unique<detail::ByteReaderBuf>(*this);
        stream_    = std::make_unique<std::istream>(streambuf_.get()); // raw ptr stays valid — same lifetime
    }
    return *stream_;
}

} // namespace ttm::plugins
