#ifndef AES_CTR_RBG_H
#define AES_CTR_RBG_H

#include <cryptopp/secblock.h>
using CryptoPP::AlignedSecByteBlock;
using CryptoPP::FixedSizeSecBlock;

#include <cryptopp/smartptr.h>
using CryptoPP::member_ptr;

#include <cryptopp/osrng.h>
using CryptoPP::OS_GenerateRandomBlock;
using CryptoPP::RandomNumberGenerator;

#include <cryptopp/aes.h>
using CryptoPP::AES;

#include <cryptopp/ccm.h>
using CryptoPP::CTR_Mode;

#include <cryptopp/sha.h>
using CryptoPP::SHA512;

#include <cryptopp/misc.h>
using CryptoPP::NotCopyable;

#include <cryptopp/config_int.h>
using CryptoPP::lword;

#include <thread>

static const long long kReseedInterval = 1LL << 48;
static const int kRandomBufferSize = BUFFER_SIZE;

class AesCtrRbg : public RandomNumberGenerator, public NotCopyable
{
public:
    explicit AesCtrRbg(const CryptoPP::byte *seed = nullptr, size_t length = 0)
    : m_pCipher(new CTR_Mode<AES>::Encryption) {
        initialised = false;
        EntropyHelper(seed, length, true);
    }

    ~AesCtrRbg() override {
        delete[] current_buffer;
        delete[] unused_buffer;
    }

    [[nodiscard]] bool CanIncorporateEntropy() const override {
        return true;
    }

    /**\brief Reseed the generator
     * @param input provided seed
     * @param length should be at least 32 for AES-128
     */
    void IncorporateEntropy(const CryptoPP::byte *input, size_t length) override {
        EntropyHelper(input, length, false);
    }

    /**Does not keep track of whether the cipher has to be reseeded. Therefore, this must be done outside of this class.
     * @param output the buffer where the block should be stored
     * @param size the number of bytes to generate
     */
    void GenerateBlock(CryptoPP::byte *output, size_t size) override
    {
        Initialise();
        size_t remaining_bytes = size;
        size_t transferred_bytes = 0;
        size_t bytes_in_buffer = kRandomBufferSize - buffer_position;
        while (bytes_in_buffer < remaining_bytes) {
            ::memcpy(output + transferred_bytes, current_buffer + buffer_position, bytes_in_buffer);
            transferred_bytes += bytes_in_buffer;
            remaining_bytes -= bytes_in_buffer;
            ReplenishBuffer();
            bytes_in_buffer = kRandomBufferSize;
        }
        ::memcpy(output + transferred_bytes, current_buffer + buffer_position, remaining_bytes);
        buffer_position += remaining_bytes;
    }

    CryptoPP::byte GenerateByte() override {
        Initialise();
        if (buffer_position == kRandomBufferSize) {
            ReplenishBuffer();
        }
        CryptoPP::byte byte = current_buffer[buffer_position];
        buffer_position += 1;
        return byte;
    }

    uint64_t GenerateUint64() {
        Initialise();
        if (buffer_position+8 > kRandomBufferSize) {
            ReplenishBuffer();
        }
        uint64_t random = *(uint64_t *)(current_buffer +buffer_position);
        buffer_position += 8;
        return random;
    }

    /**\brief makes sure that everything is initialised
     *
     */
    void Initialise() {
        if (!initialised) {
            m_pCipher->SetKeyWithIV(m_key, m_key.size(), m_iv, m_iv.size());
            current_buffer = new CryptoPP::byte[kRandomBufferSize];
            unused_buffer = new CryptoPP::byte[kRandomBufferSize];
            FillUnusedBuffer();
            ReplenishBuffer();
            initialised = true;
        }
    }

protected:
    // Sets up to use the cipher. It's a helper to allow a throw
    //   in the constructor during initialization.
    void EntropyHelper(const CryptoPP::byte* input, size_t length, bool ctor = false) {
        if(ctor)
        {
            memset(m_key, 0x00, m_key.size());
            memset(m_iv, 0x00, m_iv.size());
        }
        // 16-byte key, 16-byte nonce
        AlignedSecByteBlock seed(16 + 16);
        SHA512 hash;
        if(input && length)
        {
            // Use the user supplied seed.
            hash.Update(input, length);
        }
        else
        {
            // No seed or size. Use the OS to gather entropy.
            OS_GenerateRandomBlock(false, seed, seed.size());
            hash.Update(seed, seed.size());
        }
        hash.Update(m_key.data(), m_key.size());
        hash.Update(m_iv.data(), m_iv.size());
        hash.TruncatedFinal(seed.data(), seed.size());
        memcpy(m_key.data(), seed.data() + 0, 16);
        memcpy(m_iv.data(), seed.data() + 16, 16);
        initialised = false;
    }

private:
    std::thread buffer_thread{};
    CryptoPP::byte* current_buffer;
    CryptoPP::byte* unused_buffer;
    size_t buffer_position = 0;
    FixedSizeSecBlock<CryptoPP::byte, 16> m_key;
    FixedSizeSecBlock<CryptoPP::byte, 16> m_iv;
    member_ptr<CTR_Mode<AES>::Encryption> m_pCipher;
    bool initialised;

    void ReplenishBuffer() {
        if (buffer_thread.joinable()) {
            buffer_thread.join();
        }
        CryptoPP::byte* swap = current_buffer;
        current_buffer = unused_buffer;
        unused_buffer = swap;
        buffer_position = 0;
        buffer_thread = thread(&AesCtrRbg::FillUnusedBuffer, this);
    }

    void FillUnusedBuffer() {
        m_pCipher->GenerateBlock(unused_buffer, kRandomBufferSize);
    }
};

#endif // AES_CTR_RBG_H
