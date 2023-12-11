//
// Created by Mete Akgun on 03.07.20.
//

#ifndef PML_PARTY_H
#define PML_PARTY_H


#include <stdio.h>
#include <string.h> //strlen
#include <errno.h>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
using namespace std;

#include "../utils/constant.h"
#include "../utils/flib.h"
#include "../utils/connection.h"
#include "../utils/AesCtrRbg.h"
#include <cryptopp/osrng.h>
using CryptoPP::OS_GenerateRandomBlock;

class Party {
public:

    explicit Party(Role r, uint16_t hport=7777, const string hip="127.0.0.1", uint16_t cport=8888, const string cip="127.0.0.1", int shift = FRACTIONAL_BITS) {
        n_bits = shift;
        neg_n_bits = shift;
        p_role = r;
        if (p_role == proxy1) {
            ConnectToHelper(hip, hport, socket_helper);
            OpenP0(cip, cport, socket_p1);
        } else if (p_role == proxy2) {
            ConnectToHelper(hip, hport + SOCKET_NUMBER, socket_helper);
            ConnectToP0(cip, cport, socket_p0);
        } else if (p_role == helper) {
            OpenHelper(hip, hport, socket_p0, socket_p1);
        }
        InitialiseRbg();

        // pre-compute the truncation mask for negative values based on FRACTIONAL_BITS
        neg_truncation_mask = ((1UL << shift) - 1) << (L_BIT - shift);

        // compute the number of bits for BenchmarkExp
        // for positive power
        bool flag = false;
        int used_bits = 0;
        // for negative power
        bool neg_flag = false;
        int zero_bits = 0;
        for(int i = 7; i >= 0; i--) {
            // positive power
            int n_bits_by_i = ceil(log2(exp(pow(2, i))));
            if(L_BIT - 1 - 2 * shift - n_bits_by_i - used_bits >= 0) {
                used_bits += n_bits_by_i;
                max_power += pow(2, i);
            }
            if(!flag && L_BIT - 2 - 2 * shift - n_bits_by_i >= 0) {
                flag = true;
                n_bits += i + 1;
            }

            // negative power
            int n_zero_bits_by_i = abs(floor(log2(exp(pow(2, i))))); // how many bits are zero after the comma
            if(zero_bits + n_zero_bits_by_i < shift - 1) {
                zero_bits += n_zero_bits_by_i;
                min_power -= pow(2, i);

                if(!neg_flag) {
                    neg_flag = true;
                    neg_n_bits += i + 1;
                }
            }
        }
    }

    ~Party() {
        if (p_role == proxy1) {
            CloseSocket(socket_helper);
            CloseSocket(socket_p1);
        } else if (p_role == proxy2) {
            CloseSocket(socket_helper);
            CloseSocket(socket_p0);
        } else if (p_role == helper) {
            CloseSocket(socket_p0);
            CloseSocket(socket_p1);
        }
    }

    void InitialiseRbg() {
        if (p_role == proxy1) {
            OS_GenerateRandomBlock(false, buffer, 32);
            Send(&socket_p1[0], buffer, 32);
            common_rbg = new AesCtrRbg(buffer, 32);
            common_rbg->Initialise();
        } else if (p_role == proxy2) {
            Receive(&socket_p0[0], buffer, 32);
            common_rbg = new AesCtrRbg(buffer, 32);
            common_rbg->Initialise();
        }
        rbg = new AesCtrRbg();
        rbg->Initialise();
    }


    uint64_t GenerateRandom() {
        return rbg->GenerateUint64();
    }

    uint8_t GenerateRandomByte() {
        return rbg->GenerateByte();
    }

    uint64_t GenerateCommonRandom() {
        return common_rbg->GenerateUint64();
    }

    uint8_t GenerateCommonRandomByte() {
        return common_rbg->GenerateByte();
    }

    uint64_t CreateShare(uint64_t val){
        uint64_t share;
        if (p_role == proxy1) {
            share = GenerateCommonRandom();
        }
        else{
            share = val - GenerateCommonRandom();
        }
        return share;
    }

    uint64_t CreateShare(double val, size_t shift=FRACTIONAL_BITS){
        uint64_t v = ConvertToUint64(val, shift);
        uint64_t share;
        if (p_role == proxy1) {
            share = GenerateCommonRandom();
        }
        else{
            share = v - GenerateCommonRandom();
        }
        return share;
    }

    uint64_t* CreateShare(double *val, uint32_t sz, size_t shift = FRACTIONAL_BITS){
        uint64_t *v = ConvertToUint64(val, sz, shift);
        uint64_t *share = new uint64_t[sz];
        for (uint32_t i=0;i<sz;i++){
            if (p_role == proxy1) {
                share[i] = GenerateCommonRandom();
            }
            else{
                share[i] = v[i] - GenerateCommonRandom();
            }
        }
        delete[] v;
        return share;
    }

    uint64_t** CreateShare(double **val, uint32_t n_row, uint32_t n_col, size_t shift = FRACTIONAL_BITS){
        uint64_t **v = ConvertToUint64(val, n_row, n_col);
        uint64_t **share = new uint64_t*[n_row];
        for (uint32_t i = 0; i < n_row; i++){
            share[i] = new uint64_t[n_col];
            for(uint32_t j = 0; j < n_col; j++) {
                if (p_role == proxy1) {
                    share[i][j] = GenerateCommonRandom();
                }
                else{
                    share[i][j] = v[i][j] - GenerateCommonRandom();
                }
            }
        }
        delete[] v;
        return share;
    }

    int ReadByte() {
        if (p_role == helper) {
            ReceiveSingular(socket_p0[0], buffer, 1);
            return (int) buffer[0];
        } else
            return -1;
    }

    uint32_t ReadInt() {
        if (p_role == helper) {
            ReceiveSingular(socket_p0[0], buffer, 4);
            unsigned char *ptr = &buffer[0];
            return ConvertToInt(&ptr);
        } else
            return 0;
    }

    /**
     * Sends at least the operational Operation to the helper.
     * Additional parameters for calling the according operation might be send.
     * @param o the operation to be performed, the operation is one of the building blocks defined as constants in constant.h
     * @param params additional parameters to be send.
     * By default Null, but if additional parameters are desired to send, use a vector holding all those parameters.
     * @param size the size of params. Default is 0 but if params is not NULL, size matches the number of parameters stored in params.
     */
    void SendBytes(Operation o, uint32_t *params = nullptr, uint32_t size = 0) {
        // this will be called by both parties proxy1 and proxy2 but must only be executed for one of those,
        // otherwise, the helper will receive the same information twice (or in the worst case not read from proxy2 and mess up future results)
        if (p_role == proxy1) {
            unsigned char *ptr = &buffer[0];
            size_t s = 1;
            AddValueToCharArray((uint8_t) o, &ptr);
            if (params != nullptr && size > 0) {
                for(uint32_t i = 0; i<size; i++){
                    AddValueToCharArray((uint32_t) params[i], &ptr);
                    s += 4; // one 32 bit value requires 2^4 bits; if we were to store 64 bit values --> += 8
                }
            }
            SendSingular(socket_helper[0], buffer, s);
        }
    }

    static void PrintPaperFriendly(double time_taken) {
        cout << "Paper\t" << (bytes_sent / 1e6) << "\t" << (bytes_received / 1e6) << "\t" << fixed << time_taken << setprecision(9) << endl;
    }

    [[nodiscard]] Role GetPRole() const {
        return p_role;
    }

    [[nodiscard]] int *GetSocketP1(){
        return socket_p0;
    }

    [[nodiscard]] int *GetSocketP2(){
        return socket_p1;
    }

    [[nodiscard]] int *GetSocketHelper(){
        return socket_helper;
    }

    [[nodiscard]] uint8_t *GetBuffer1(){
        return buffer;
    }

    [[nodiscard]] uint8_t *GetBuffer2(){
        return buffer2;
    }

    [[nodiscard]] int GetNBits() const {
        return n_bits;
    }

    void SetNBits(int nBits) {
        n_bits = nBits;
    }

    [[nodiscard]] double GetMaxPower() const {
        return max_power;
    }

    void SetMaxPower(double maxPower) {
        max_power = maxPower;
    }

    [[nodiscard]] int GetNegNBits() const {
        return neg_n_bits;
    }

    void SetNegNBits(int negNBits) {
        neg_n_bits = negNBits;
    }

    [[nodiscard]] double GetMinPower() const {
        return min_power;
    }

    void SetMinPower(double minPower) {
        min_power = minPower;
    }

    [[nodiscard]] uint64_t GetNegTruncationMask() const {
        return neg_truncation_mask;
    }
private:
    Role p_role;
    int socket_p0[SOCKET_NUMBER],socket_p1[SOCKET_NUMBER],socket_helper[SOCKET_NUMBER];
    uint8_t buffer[BUFFER_SIZE];
    uint8_t buffer2[BUFFER_SIZE];
    AesCtrRbg* common_rbg;
    AesCtrRbg* rbg;
    int n_bits; // number of bits of a value to consider in the BenchmarkExp computation
    int neg_n_bits; // number of bits of a negative value to consider in the BenchmarkExp computation
    double max_power = 0;
    double min_power = 0;
    uint64_t neg_truncation_mask = 0;
};



#endif //PML_PARTY_H
