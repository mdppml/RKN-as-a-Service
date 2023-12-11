//
// Created by aburak on 24.03.22.
//

#ifndef CECILIA_RKN_H
#define CECILIA_RKN_H

#include "core.h"
#include "../utils/flib.h"
#include "../dependencies/eigen/Eigen/Eigen"

using namespace Eigen;

uint64_t** RandomOrthogonalMatrix(Party* proxy, uint32_t size) {
    /*
     * Generate a size-by-size random unit-length orthogonal matrix in proxy1 and matrix of zeros in proxy2
     * The solution is based on this: https://scicomp.stackexchange.com/a/34974
     *
     * Input(s)
     * size: the dimension of the matrix, which is size-by-size
     *
     * Output(s)
     * Returns a size-by-size matrix
     */
    if(proxy->GetPRole() == proxy1 || proxy->GetPRole() == proxy2) {
        // generate a vector of random values
        double *rand_vals = new double[size * size];
        for (uint32_t i = 0; i < size * size; i++) {
            // I guess there is no need for both proxies to know the orthogonal matrix
            rand_vals[i] = ConvertToDouble(proxy->GenerateCommonRandom() & ORTHOGONAL_MASK);
        }

        // create an orthogonal matrix by using the generated random array in Eigen - the result is a Matrix
        Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(rand_vals, size, size);
        Matrix<double, Dynamic, Dynamic, RowMajor> XtX = X.transpose() * X;
        SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic, RowMajor>> es(XtX);
        Matrix<double, Dynamic, Dynamic, RowMajor> S = es.operatorInverseSqrt();
        Matrix<double, Dynamic, Dynamic, RowMajor> orth = X * S;

        // convert a Matrix of double into two dimensional uint64_t array
        double *tmp = new double[size * size];
        Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(tmp, size, size) = orth;
        uint64_t **M = new uint64_t *[size];
        double** debugging_M = new double*[size];
        for (int i = 0; i < size; i++) {
            M[i] = new uint64_t[size];
            debugging_M[i] = new double[size];
            for (int j = 0; j < size; j++) {
                M[i][j] = ConvertToUint64(tmp[i * size + j]);
                debugging_M[i][j] = tmp[i * size + j];
            }
        }
        return M;
    }
    return nullptr;
}

void EigenDecomposition(Party *proxy, uint32_t size, double epsilon = 0.01) {
    /*
     * Perform eigenvalue decomposition of a single Gram matrix
     */
    int p_role = proxy->GetPRole();
    int *socket_p1 = proxy->GetSocketP1();
    int *socket_p2 = proxy->GetSocketP2();

    if (p_role == helper) {
        // receive the shares of the masked Gram matrix
        thread thr1 = thread(Receive, socket_p1, proxy->GetBuffer1(), size * size * 8);
        thread thr2 = thread(Receive, socket_p2, proxy->GetBuffer2(), size * size * 8);
        thr1.join();
        thr2.join();

        unsigned char *ptr = proxy->GetBuffer1();
        unsigned char *ptr2 = proxy->GetBuffer2();

        uint64_t **G1;
        uint64_t **G2;
        ConvertTo2dArray(&ptr, G1, size, size);
        ConvertTo2dArray(&ptr2, G2, size, size);

        // perform eigenvalue decomposition
        // 1. convert to double array
        double *masked_G = new double[size * size];
        for (uint32_t i = 0; i < size * size; i++) {
            masked_G[i] = ConvertToDouble(G1[i / size][i % size] + G2[i / size][i % size]);
        }

        // 2. convert to Matrix
        Map<Matrix<double, Dynamic, Dynamic, RowMajor>> masked_matrix_G(masked_G, size,
                                                                        size); // make sure that this conversion preserves the same orientation - row-major vs column-major!!!

        // 3. perform eigenvalue decomposition
        EigenSolver<Matrix<double, Dynamic, Dynamic, RowMajor>> eig_solver;
        eig_solver.compute(masked_matrix_G);
        ptr = proxy->GetBuffer1();
        ptr2 = proxy->GetBuffer2();

        // eigenvector - size-by-size
        if(DEBUG_FLAG >= 2)
            cout << "Step 3.1: eigenvectors" << endl;
        Matrix<double, Dynamic, Dynamic, RowMajor> eig_vectors = eig_solver.eigenvectors().real();
        double *matrix2double = new double[size * size];
        Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(matrix2double, size,
                size) = eig_vectors; // convert from Matrix to double*
        uint64_t **M1 = new uint64_t *[size];
        uint64_t **M2 = new uint64_t *[size];
        uint64_t tmp_share;
        for (int i = 0; i < size; i++) {
            M1[i] = new uint64_t[size];
            M2[i] = new uint64_t[size];
            for (int j = 0; j < size; j++) {
                tmp_share = proxy->GenerateRandom();
                M1[i][j] = ConvertToUint64(matrix2double[i * size + j]) -
                           tmp_share; // make sure that this conversion preserves the same orientation - row-major vs column-major!!!
                M2[i][j] = tmp_share;
                AddValueToCharArray(M1[i][j], &ptr);
                AddValueToCharArray(M2[i][j], &ptr2);
            }
        }

        // eigenvalues - size-by-1
        double *double_eig_values = new double[size];
        Matrix<double, Dynamic, 1> vec_eig_values = eig_solver.eigenvalues().real();
        Map<Matrix<double, Dynamic, 1>>(double_eig_values, size,
                1) = vec_eig_values; // Since it is a vector, do we need to specify that it is row-major?

//        if(DEBUG_FLAG >= 3)
        Print1dArray("Lambda + s", double_eig_values, size);

//        double* delta = new double[size]; // a vector of random values to mask eigenvalues
        uint64_t *masked_eig_values = new uint64_t[size];
        double alpha = MIN_DELTA + ((double) rand() / RAND_MAX) * (MAX_DELTA - MIN_DELTA); // a scalar random value to mask eigenvalues
        double* delta = new double[size];
        for (uint32_t i = 0; i < size; i++) {
            delta[i] = MIN_DELTA + ((double) rand() / RAND_MAX) * (MAX_DELTA - MIN_DELTA);
            masked_eig_values[i] = ConvertToUint64((double_eig_values[i] + epsilon) * delta[i] + alpha);
            AddValueToCharArray(masked_eig_values[i], &ptr);
            AddValueToCharArray(ConvertToUint64(delta[i]), &ptr2);
        }

        AddValueToCharArray(ConvertToUint64(alpha), &ptr2); // add scalar alpha to buffer2

        // 4. send
        //  - the share of eigenvectors (size * size * 8 bits) and masked eigenvalues (size * 8 bits) to proxy1
        //  - the share of eigenvectors (size * size * 8 bits), the mask delta (size * 8 bits) and scalar alpha (8 bits) to proxy2
        thr1 = thread(Send, socket_p1, proxy->GetBuffer1(), size * size * 8 + size * 8);
        thr2 = thread(Send, socket_p2, proxy->GetBuffer2(), size * size * 8 + size * 8 + 8);
        thr1.join();
        thr2.join();

        for (uint32_t i = 0; i < size; i++) {
            delete[] M1[i];
            delete[] M2[i];
        }
        delete[] M1;
        delete[] M2;
        delete[] double_eig_values;
        delete[] masked_eig_values;
        delete[] matrix2double;
        delete[] masked_G;
        delete[] delta;
    }
}

void EigenDecomposition(Party* proxy, uint32_t n_gms, uint32_t size, double epsilon = 0.01) {
    /*
     * Perform eigenvalue decomposition of a single Gram matrix
     */
    int p_role = proxy->GetPRole();
    int *socket_p1 = proxy->GetSocketP1();
    int *socket_p2 = proxy->GetSocketP2();

    if (p_role == helper) {
        // receive the shares of the masked Gram matrix
        thread thr1 = thread(Receive, socket_p1, proxy->GetBuffer1(), n_gms * size * size * 8);
        thread thr2 = thread(Receive, socket_p2, proxy->GetBuffer2(), n_gms * size * size * 8);
        thr1.join();
        thr2.join();

        unsigned char *ptr = proxy->GetBuffer1();
        unsigned char *ptr2 = proxy->GetBuffer2();

        uint64_t ***G1;
        uint64_t ***G2;
        ConvertTo3dArray(&ptr, G1, n_gms, size, size);
        ConvertTo3dArray(&ptr2, G2, n_gms, size, size);

        // perform eigenvalue decomposition
        // 1. convert to double array
        double **masked_G = new double*[n_gms];
        for(uint32_t g = 0; g < n_gms; g++) {
            masked_G[g] = new double[size * size];
            for (uint32_t i = 0; i < size * size; i++) {
                masked_G[g][i] = ConvertToDouble(G1[g][i / size][i % size] + G2[g][i / size][i % size]);
            }
        }

        // initialize pointers to the beginning of the buffers to send eigenvalue- and eigenvector-related things
        ptr = proxy->GetBuffer1();
        ptr2 = proxy->GetBuffer2();

        // pre-allocation
        double *matrix2double = new double[size * size];
        uint64_t **M1 = new uint64_t *[size];
        uint64_t **M2 = new uint64_t *[size];
        for (int i = 0; i < size; i++) {
            M1[i] = new uint64_t[size];
            M2[i] = new uint64_t[size];
        }
        double *double_eig_values = new double[size];
        uint64_t *masked_eig_values = new uint64_t[size];
        double* delta = new double[size];
        uint64_t** dbg_masked_eig_values = new uint64_t*[n_gms];

        // for each gram matrix
        for(uint32_t g = 0; g < n_gms; g++) {
            // 2. convert to Matrix
            Map<Matrix<double, Dynamic, Dynamic, RowMajor>> masked_matrix_G(masked_G[g], size,
                                                                            size); // make sure that this conversion preserves the same orientation - row-major vs column-major!!!

            // 3. perform eigenvalue decomposition
            EigenSolver<Matrix<double, Dynamic, Dynamic, RowMajor>> eig_solver;
            eig_solver.compute(masked_matrix_G);

            // eigenvector - size-by-size
            Matrix<double, Dynamic, Dynamic, RowMajor> eig_vectors = eig_solver.eigenvectors().real();

            Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(matrix2double, size,
                    size) = eig_vectors; // convert from Matrix to double*

            uint64_t tmp_share;
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    tmp_share = proxy->GenerateRandom();
                    M1[i][j] = ConvertToUint64(matrix2double[i * size + j]) -
                               tmp_share; // make sure that this conversion preserves the same orientation - row-major vs column-major!!!
                    M2[i][j] = tmp_share;
                    AddValueToCharArray(M1[i][j], &ptr);
                    AddValueToCharArray(M2[i][j], &ptr2);
                }
            }

            // eigenvalues - size-by-1
            Matrix<double, Dynamic, 1> vec_eig_values = eig_solver.eigenvalues().real();
            Map<Matrix<double, Dynamic, 1>>(double_eig_values, size,
                    1) = vec_eig_values; // Since it is a vector, do we need to specify that it is row-major?
            double alpha = MIN_DELTA + ((double) rand() / RAND_MAX) * (MAX_DELTA - MIN_DELTA); // a scalar random value to mask eigenvalues
            for (uint32_t i = 0; i < size; i++) {
                delta[i] = MIN_DELTA + ((double) rand() / RAND_MAX) * (MAX_DELTA - MIN_DELTA);
                masked_eig_values[i] = ConvertToUint64((double_eig_values[i] + epsilon) * delta[i] + alpha);
                AddValueToCharArray(masked_eig_values[i], &ptr);
                AddValueToCharArray(ConvertToUint64(delta[i]), &ptr2);
            }

            dbg_masked_eig_values[g] = masked_eig_values;

            AddValueToCharArray(ConvertToUint64(alpha), &ptr2); // add scalar alpha to buffer2
        }

        // 4. send
        //  - the share of eigenvectors (size * size * 8 bits) and masked eigenvalues (size * 8 bits) to proxy1
        //  - the share of eigenvectors (size * size * 8 bits), the mask delta (size * 8 bits) and scalar alpha (8 bits) to proxy2
        thr1 = thread(Send, socket_p1, proxy->GetBuffer1(), n_gms * (size * size * 8 + size * 8));
        thr2 = thread(Send, socket_p2, proxy->GetBuffer2(), n_gms * (size * size * 8 + size * 8 + 8));
        thr1.join();
        thr2.join();

        for (uint32_t i = 0; i < size; i++) {
            delete[] M1[i];
            delete[] M2[i];
        }
        delete[] M1;
        delete[] M2;
        delete[] double_eig_values;
        delete[] masked_eig_values;
        delete[] matrix2double;
        delete[] masked_G;
        delete[] delta;
    }
}

uint64_t*** GaussianKernel(Party* proxy, uint64_t ***G, uint64_t alpha, uint32_t n_gms, uint32_t size) {
    /* This function computes the Gaussian kernel matrices based on the Gram matrices of the samples in bulk form.
     *
     * Input(s)
     * G: n_gms-many size-by-size Gram matrices
     * alpha: the similarity adjustment parameter of the Gaussian kernel function
     * n_gms: number of Gram matrices
     * size: number of samples in the Gram matrices
     *
     * Output(s)
     * Returns n_gms-many Gaussian kernel matrices of size size-by-size
     */
    int p_role = proxy->GetPRole();
    if(p_role == proxy1 || p_role == proxy2) {
        uint32_t step_size = (size * (size + 1)) / 2;

        // computes the initial form of the kernel matrices
        uint64_t ***km = new uint64_t**[n_gms];
        // initialization
        for(uint32_t g = 0; g < n_gms; g++) {
            km[g] = new uint64_t*[size];
            for(uint32_t k = 0; k < size; k++) {
                km[g][k] = new uint64_t[size];
            }
        }

        // straighten the initial kernel matrix
        uint64_t *str_km = new uint64_t[n_gms * step_size];
        uint32_t ind = 0;
        for(uint32_t g = 0; g < n_gms; g++) {
            for(uint32_t i = 0; i < size; i++) {
                for(uint32_t j = i; j < size; j++) {
                    str_km[ind] = LocalMultiply(alpha, (G[g][i][j] - ((uint64_t) 1 << FRACTIONAL_BITS) * p_role));
                    ind++;
                }
            }
        }

        // compute the BenchmarkExp of the values in the kernel matrix
        uint64_t *tmp_exp = Exp(proxy, str_km, n_gms * step_size);

        cout << "GaussianKernel: Exp computation is done!" << endl;

        // the first k layer is the first part of the resulting tmp_exp
        ind = 0;
        for(int i = 0; i < size; i++) {
            for(int j = i; j < size; j++) {
                km[0][i][j] = tmp_exp[ind];
                km[0][j][i] = km[0][i][j];
                ind++;
            }
        }

        // multiplication of the individually computed kernel matrices and the restoration of these multiplications
        uint64_t *tmp_mul = tmp_exp;
        for (int g = 1; g < n_gms; g++) {
            tmp_mul = Multiply(proxy, tmp_mul, &tmp_exp[g * step_size], step_size);
            ind = 0;
            for(int i = 0; i < size; i++) {
                for(int j = i; j < size; j++) {
                    km[g][i][j] = tmp_mul[ind];
                    km[g][j][i] = km[g][i][j];
                    ind++;
                }
            }
        }
        return km;
    }
    else if(p_role == helper) {
        uint32_t step_size = (size * (size + 1)) / 2;

        Exp(proxy, nullptr, n_gms * step_size);
        for (int g = 1; g < n_gms; g++) {
            Multiply(proxy, nullptr, nullptr, step_size);
        }
        return nullptr;
    }
    return nullptr;
}

uint64_t** InverseSqrt(Party* proxy, uint64_t **G, uint32_t size, double epsilon = 0.01) {
    /*
     * Computes the inverse square root of the given size-by-size gram matrix G by employing eigenvalue decomposition.
     * We based our solution ma
     * Reference: Zhou, Lifeng, and Chunguang Li. "Outsourcing eigen-decomposition and singular value decomposition of
     * large matrix to a public cloud." IEEE Access 4 (2016): 869-879.
     *
     * Input(s)
     * G: size-by-size gram matrix
     * size: the size of the gram matrix
     *
     * Output(s)
     *
     */
    int p_role = proxy->GetPRole();
    if (p_role == proxy1 || p_role == proxy2) {
        // generate a scalar value whose max is MAX_SCALAR
        uint64_t scalar_s = proxy->GenerateCommonRandom() & MAX_SCALAR;
        PrintValue("s", ConvertToDouble(scalar_s));
        uint64_t scalar_a = proxy->GenerateCommonRandom() & MAX_A; // multiplier
        double d_scalar_a = ConvertToDouble(scalar_a);
        PrintValue("a", d_scalar_a);
        // v2: compute A1 + sI
        for( int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                G[i][j] = LocalMultiply(G[i][j], scalar_a);
                if(i == j && p_role == proxy1) {
                    G[i][i] += scalar_s;
                }
            }
        }
        cout << "C1" << endl;
        // generate a random orthogonal matrix
        uint64_t **M = RandomOrthogonalMatrix(proxy, size);
        cout << "C2" << endl;
        // compute M * (A + sI) * M^T
        uint64_t **masked_gram_matrix = LocalMatrixMatrixMultiply(M, G, size, size, size);
        cout << "C3" << endl;
        uint64_t **trM;
        trM = new uint64_t *[size];
        for (int i = 0; i < size; i++) {
            trM[i] = new uint64_t[size];
            for (int j = 0; j < size; j++) {
                trM[i][j] = M[j][i];
            }
        }
        masked_gram_matrix = LocalMatrixMatrixMultiply(masked_gram_matrix, trM, size, size, size);
        cout << "C4" << endl;
        // send the mask Gram matrix to Helper
        EigenDecomposition(proxy, size, epsilon);
        cout << "C4.5" << endl;
        unsigned char *ptr = proxy->GetBuffer1();
        AddArrayToCharArray(masked_gram_matrix, &ptr, size, size);
        Send(proxy->GetSocketHelper(), proxy->GetBuffer1(), size * size * 8);

        /* receive the corresponding part of the resulting eigenvalue decomposition from Helper - note that these are
        specific for the computation of the inverse square root of the Gram matrix */
        // First size * size * 8 bits: masked eigenvalues (or unmasker in case proxy2)
        // Second size * size * 8 bits: masked eigenvectors
        // Additional 8 bits to get the scalar alpha in proxy2
        Receive(proxy->GetSocketHelper(), proxy->GetBuffer1(), size * size * 8 + size * 8 + (8 * p_role));
        if(DEBUG_FLAG >= 2)
            cout << "Received the components of eigenvalue decomposition" << endl;
        ptr = proxy->GetBuffer1();
        uint64_t alpha;
        uint64_t **masked_eig_vecs;
        uint64_t *eig_vals;
        cout << "C4.6" << endl;
        ConvertTo2dArray(&ptr, masked_eig_vecs, size, size); // the share of the masked eigenvectors
        cout << "C4.7" << endl;
        ConvertToArray(&ptr, eig_vals, size); // eigenvalue related things
        cout << "C4.8" << endl;

        Print1dArray("Received eigenvalue related things", ConvertToDouble(eig_vals, size), size);

        if (p_role == proxy2) {
            alpha = ConvertToLong(&ptr); // scalar alpha
            if(DEBUG_FLAG >= 3)
                PrintValue("Alpha", ConvertToDouble(alpha));
        }

        // unmasking the eigenvectors
        uint64_t **eig_vecs = LocalMatrixMatrixMultiply(trM, masked_eig_vecs, size, size, size);

        if(DEBUG_FLAG >= 3)
            Print2dArray("Reconstructed eigenvectors",
                         ConvertToDouble(Reconstruct(proxy, eig_vecs, size, size), size, size), size, size);

        // unmasking the inverse square root of the eigenvalues
        uint64_t *unmasker = new uint64_t[size];
        if (p_role == proxy1) {
            if(DEBUG_FLAG >= 2)
                cout << "Receiving unmasker from proxy2..." << endl;
            Receive(proxy->GetSocketP2(), proxy->GetBuffer1(), size * 8);
            if(DEBUG_FLAG >= 2)
                cout << "Done!" << endl;
            ptr = proxy->GetBuffer1();
            ConvertToArray(&ptr, unmasker, size);
            for (uint32_t i = 0; i < size; i++) {
                eig_vals[i] = ConvertToUint64(ConvertToDouble(eig_vals[i]) / d_scalar_a) - unmasker[i];
            }
        } else {
            for (uint32_t i = 0; i < size; i++) {
                unmasker[i] = ConvertToUint64(ConvertToDouble(LocalMultiply(scalar_s, eig_vals[i]) + alpha) / d_scalar_a);
            }

            if(DEBUG_FLAG >= 3)
                Print1dArray("Unmasker", ConvertToDouble(unmasker, size), size);

            ptr = proxy->GetBuffer1();
            AddValueToCharArray(unmasker, &ptr, size);
            if(DEBUG_FLAG >= 2)
                cout << "Sending the unmasker to proxy1..." << endl;
            Send(proxy->GetSocketP1(), proxy->GetBuffer1(), size * 8);
            if(DEBUG_FLAG >= 2)
                cout << "Done!" << endl;
        }

        uint64_t *zero_vec = new uint64_t[size];
        for (uint32_t i = 0; i < size; i++) {
            zero_vec[i] = 0;
            if(p_role == proxy1) {
                eig_vals[i] = ConvertToUint64(1.0 / pow(ConvertToDouble(eig_vals[i]), 0.5));
            }
            else {
                eig_vals[i] = ConvertToUint64(pow(ConvertToDouble(eig_vals[i]), 0.5));
            }
        }

        if(DEBUG_FLAG >= 3)
            Print1dArray("Revised eigenvalue related things", ConvertToDouble(eig_vals, size), size);

        if(DEBUG_FLAG >= 2)
            cout << "Computing the inverse square root of the eigenvalues..." << endl;
        uint64_t *sqrt_eig_vals;
        if (p_role == proxy1) {
            sqrt_eig_vals = Multiply(proxy, eig_vals, zero_vec, size);
        } else {
            sqrt_eig_vals = Multiply(proxy, zero_vec, eig_vals, size);
        }

        // if(DEBUG_FLAG >= 3)
        Print1dArray("Inverse square root of the eigenvalues",
                     ConvertToDouble(Reconstruct(proxy, sqrt_eig_vals, size), size), size);

        if(DEBUG_FLAG >= 2)
            cout << "The inverse square root of the eigenvalues are computed." << endl;

        // construct the inverse square root of the Gram matrix
        uint64_t **tr_eig_vecs = new uint64_t *[size];
        uint64_t **eig_vals_mat = new uint64_t *[size];
        for (uint32_t i = 0; i < size; i++) {
            tr_eig_vecs[i] = new uint64_t[size];
            eig_vals_mat[i] = new uint64_t[size];
            eig_vals_mat[i][i] = sqrt_eig_vals[i];
            for (uint32_t j = 0; j < size; j++) {
                tr_eig_vecs[i][j] = eig_vecs[j][i];

                // if the elements of the two-dimensional dynamic array are zero after the initialization, no need for this part
                if (j != i) {
                    eig_vals_mat[i][j] = 0;
                }
            }
        }
        cout << "C5" << endl;
        uint64_t **invsqrt_G = MatrixMatrixMultiply(proxy,
                                                    MatrixMatrixMultiply(proxy, eig_vecs, eig_vals_mat, size, size,
                                                                         size),
                                                    tr_eig_vecs, size, size, size);
        if(DEBUG_FLAG >= 1)
            cout << "Returning from Party::InverseSqrt...\n************************************************************" << endl;
        return invsqrt_G;
    }
    else if(p_role == helper) {
        EigenDecomposition(proxy, size);
        Multiply(proxy, nullptr, nullptr, size);
        MatrixMatrixMultiply(proxy, nullptr, nullptr, size * size * size, 0, 0);
        MatrixMatrixMultiply(proxy, nullptr, nullptr, size * size * size, 0, 0);
        return nullptr;
    }
    return nullptr;
}

uint64_t*** InverseSqrt(Party* proxy, uint64_t ***G, uint32_t n_gms, uint32_t size, double epsilon = 0.01) {
    /*
     * Computes the inverse square root of the given size-by-size n_gms gram matrices in G by employing eigenvalue
     * decomposition. We based our solution the following:
     * Reference: Zhou, Lifeng, and Chunguang Li. "Outsourcing eigen-decomposition and singular value decomposition of
     * large matrix to a public cloud." IEEE Access 4 (2016): 869-879.
     *
     * Input(s)
     * G: n_gms-by-size-by-size gram matrix
     * n_gms: the number of gram matrices in G
     * size: the size of the gram matrix
     *
     * Output(s)
     *
     */
    int p_role = proxy->GetPRole();
    if (p_role == proxy1 || p_role == proxy2) {
        // generate a scalar value whose max is MAX_SCALAR
        uint64_t scalar_s = proxy->GenerateCommonRandom() & MAX_SCALAR;
        if (p_role == proxy1) {
            // compute A1 + sI
            for(int g = 0; g < n_gms; g++) {
                for (int i = 0; i < size; i++) {
                    G[g][i][i] += scalar_s;
                }
            }
        }

        // generate random orthogonal matrices
        uint64_t ***M = new uint64_t**[n_gms];
        for(int i = 0; i < n_gms; i++) {
            M[i] = RandomOrthogonalMatrix(proxy, size);
            double** tmp_M = ConvertToDouble(M[i], size, size);
            double** tr_tmp_M = new double*[size];
            for(int j = 0; j < size; j++) {
                tr_tmp_M[j] = new double[size];
                for(int k = 0; k < size; k++) {
                    tr_tmp_M[j][k] = tmp_M[k][j];
                }
            }
        }

        // compute M * (A + sI) * M^T
        uint64_t ***masked_gram_matrix = LocalMatrixMatrixMultiply(M, G, n_gms, size, size, size);

        uint64_t ***trM;
        trM = new uint64_t**[n_gms];
        for(int g = 0; g < n_gms; g++) {
            trM[g] = new uint64_t *[size];
            for (int i = 0; i < size; i++) {
                trM[g][i] = new uint64_t[size];
                for (int j = 0; j < size; j++) {
                    trM[g][i][j] = M[g][j][i];
                }
            }
        }
        masked_gram_matrix = LocalMatrixMatrixMultiply(masked_gram_matrix, trM, n_gms, size, size, size);

        #ifndef NDEBUG
            double d_scalar_s = ConvertToDouble(scalar_s);
            double*** rec_kmer_kms = new double**[n_gms];
            double*** rec_M = new double**[n_gms];
            double*** rec_tr_M = new double**[n_gms];
            for(int g = 0; g < n_gms; g++) {
                rec_kmer_kms[g] = ConvertToDouble(Reconstruct(proxy, G[g], size, size), size, size);
                rec_M[g] = ConvertToDouble(M[g], size, size);
                rec_tr_M[g] = ConvertToDouble(trM[g], size, size);
            }
            double*** MGMT = DoubleMatrixMatrixMultiply(
                    DoubleMatrixMatrixMultiply(rec_M, rec_kmer_kms, n_gms, size, size, size), rec_tr_M,
                    n_gms, size, size, size);
            double** eig_MGMT = new double*[n_gms];
            for(int i = 0; i < n_gms; i++) {
                EigenSolver<Matrix<double, Dynamic, Dynamic, RowMajor>> AT_ges;
                Map<Matrix<double, Dynamic, Dynamic, RowMajor>> AT_matrix_G(Straighten2dArray(MGMT[i], size, size), size, size);
                AT_ges.compute(AT_matrix_G);
                Matrix<double, Dynamic, 1> AT_eig_vals = AT_ges.eigenvalues().real();
                eig_MGMT[i] = new double[size];
                Map<Matrix<double, Dynamic, 1>>(eig_MGMT[i], size) = AT_eig_vals;
                for(int j = 0; j < size; j++) {
                    eig_MGMT[i][j] -= d_scalar_s;
                }
                BubbleSort(eig_MGMT[i], size);
            }
        Print2dArray("Eigenvalues of MGMT", eig_MGMT, n_gms, size, false);

            double*** rec_masked_gram_matrix = new double**[n_gms];
            for(int i = 0; i < n_gms; i++) {
                rec_masked_gram_matrix[i] = ConvertToDouble(Reconstruct(proxy, masked_gram_matrix[i], size, size), size, size);
            }

            double*** diff_MGMT = new double**[n_gms];
            double* total_diffs = new double[n_gms];

            for(int i = 0; i < n_gms; i++) {
                double tmp = 0;
                int cntr_invalid = 0;
                int cntr_negative = 0;
                diff_MGMT[i] = new double*[size];
                for(int j = 0; j < size; j++) {
                    diff_MGMT[i][j] = new double[size];
                    for(int k = 0; k < size; k++) {
                        diff_MGMT[i][j][k] = MGMT[i][j][k] - rec_masked_gram_matrix[i][j][k];
                        tmp += diff_MGMT[i][j][k];
                        if(abs(MGMT[i][j][k] - rec_masked_gram_matrix[i][j][k]) > 1) {
                            cout << "MGMT[" << i << "][" << j << "][" << k << "]: " << MGMT[i][j][k] << "\tComputed: " << rec_masked_gram_matrix[i][j][k] << endl;
                            cntr_invalid++;
                        }
                    }
                }
                total_diffs[i] = tmp;
                cout << "In " << i << "th kernel matrix, there are " << cntr_invalid << " invalid entries." << endl;
            }
        Print1dArray("Differences between GT MGMT and computed one", total_diffs, n_gms);
        #endif // NDEBUG

        // send the mask Gram matrix to Helper
        EigenDecomposition(proxy, n_gms, size, epsilon);
        unsigned char *ptr = proxy->GetBuffer1();
        for(uint32_t i = 0; i < n_gms; i++) {
            AddArrayToCharArray(masked_gram_matrix[i], &ptr, size, size);
        }
        Send(proxy->GetSocketHelper(), proxy->GetBuffer1(), n_gms * size * size * 8);

        /* receive the corresponding part of the resulting eigenvalue decomposition from Helper - note that these are
        specific for the computation of the inverse square root of the Gram matrix */
        // We have n_gms chunks. For each chunk of (size * size * 8 + size * 8 + p_role * 8) bits block:
        // First size * size * 8 bits: masked eigenvalues (or unmasker in case proxy2)
        // Second size * size * 8 bits: masked eigenvectors
        // Additional 8 bits to get the scalar alpha in proxy2
        Receive(proxy->GetSocketHelper(), proxy->GetBuffer1(), n_gms * (size * size * 8 + size * 8 + (8 * p_role)));
        ptr = proxy->GetBuffer1();
        uint64_t *alpha = new uint64_t[n_gms];
        uint64_t ***masked_eig_vecs = new uint64_t**[n_gms];
        uint64_t **eig_vals = new uint64_t*[n_gms];

        for(uint64_t g = 0; g < n_gms; g++) {
            ConvertTo2dArray(&ptr, masked_eig_vecs[g], size, size); // the share of the masked eigenvectors
            ConvertToArray(&ptr, eig_vals[g], size); // eigenvalue related things

            if (p_role == proxy2) {
                alpha[g] = ConvertToLong(&ptr); // scalar alpha
            }
        }

        // unmasking the eigenvectors
        uint64_t ***eig_vecs = LocalMatrixMatrixMultiply(trM, masked_eig_vecs, n_gms, size, size, size);

        // unmasking the inverse square root of the eigenvalues
        uint64_t **unmasker;
        if (p_role == proxy1) {
            Receive(proxy->GetSocketP2(), proxy->GetBuffer1(), n_gms * size * 8);
            ptr = proxy->GetBuffer1();
            ConvertTo2dArray(&ptr, unmasker, n_gms, size);
            for(uint32_t g = 0; g < n_gms; g++) {
                for (uint32_t i = 0; i < size; i++) {
                    eig_vals[g][i] = eig_vals[g][i] - unmasker[g][i];
                }
            }
        } else {
            ptr = proxy->GetBuffer1();
            unmasker = new uint64_t*[n_gms];
            for(uint32_t g = 0; g < n_gms; g++) {
                // s * delta + alpha
                unmasker[g] = new uint64_t[size];
                for (uint32_t i = 0; i < size; i++) {
                    unmasker[g][i] = LocalMultiply(scalar_s, eig_vals[g][i]) + alpha[g];
                }
                AddValueToCharArray(unmasker[g], &ptr, size);
            }
            Send(proxy->GetSocketP1(), proxy->GetBuffer1(), n_gms * size * 8);
        }

        uint64_t *str_zero_vec = new uint64_t[n_gms * size];
        uint64_t *str_untouched_eigvals = new uint64_t[n_gms * size]; // for debugging purposes
        uint64_t *str_processed_eigvals = new uint64_t[n_gms * size];
        for(uint32_t g = 0; g < n_gms; g++) {
            for (uint32_t i = 0; i < size; i++) {
                str_zero_vec[g * size + i] = 0;
                if(p_role == proxy1) {
                    str_untouched_eigvals[g * size + i] = eig_vals[g][i];
                    str_processed_eigvals[g * size + i] = ConvertToUint64(
                            1.0 / pow(ConvertToDouble(eig_vals[g][i]), 0.5));
                }
                else {
                    str_untouched_eigvals[g * size + i] = ConvertToUint64(1.0 / ConvertToDouble(eig_vals[g][i]));
                    str_processed_eigvals[g * size + i] = ConvertToUint64(pow(ConvertToDouble(eig_vals[g][i]), 0.5));
                }
            }
        }

        uint64_t *tmp_res;
        if (p_role == proxy1) {
            tmp_res = Multiply(proxy, str_processed_eigvals, str_zero_vec, n_gms * size);
        } else {
            tmp_res = Multiply(proxy, str_zero_vec, str_processed_eigvals, n_gms * size);
        }

        uint64_t **sqrt_eig_vals = new uint64_t*[n_gms];
        for(uint32_t g = 0; g < n_gms; g++) {
            sqrt_eig_vals[g] = new uint64_t[size];
            for(uint32_t i = 0; i < size; i++) {
                sqrt_eig_vals[g][i] = tmp_res[g * size + i];
            }
        }

        // construct the inverse square root of the Gram matrix
        uint64_t ***tr_eig_vecs = new uint64_t**[n_gms];
        uint64_t ***eig_vals_mat = new uint64_t**[n_gms];
        for(uint32_t g = 0; g < n_gms; g++) {
            tr_eig_vecs[g] = new uint64_t*[size];
            eig_vals_mat[g] = new uint64_t*[size];
            for (uint32_t i = 0; i < size; i++) {
                tr_eig_vecs[g][i] = new uint64_t[size];
                eig_vals_mat[g][i] = new uint64_t[size];
                eig_vals_mat[g][i][i] = sqrt_eig_vals[g][i];
                for (uint32_t j = 0; j < size; j++) {
                    tr_eig_vecs[g][i][j] = eig_vecs[g][j][i];

                    // if the elements of the two-dimensional dynamic array are zero after the initialization, no need for this part
                    if (j != i) {
                        eig_vals_mat[g][i][j] = 0;
                    }
                }
            }
        }

        uint64_t*** tmp_G = MatrixMatrixMultiply(proxy, eig_vecs, eig_vals_mat, n_gms, size, size, size);
        uint64_t ***invsqrt_G = MatrixMatrixMultiply(proxy, tmp_G, tr_eig_vecs, n_gms, size, size, size);

        for(uint32_t g = 0; g < n_gms; g++) {
            delete [] sqrt_eig_vals[g];
            for(uint32_t i = 0; i < size; i++) {
                delete [] tr_eig_vecs[g][i];
                delete [] eig_vals_mat[g][i];
            }
            delete [] tr_eig_vecs[g];
            delete [] eig_vals_mat[g];
        }
        delete [] sqrt_eig_vals;
        delete [] tr_eig_vecs;
        delete [] eig_vals_mat;
        delete [] str_processed_eigvals;
        delete [] str_zero_vec;
        delete [] tmp_res;

        return invsqrt_G;
    }
    else if(p_role == helper) {
        EigenDecomposition(proxy, n_gms, size);
        Multiply(proxy, nullptr, nullptr, n_gms * size);
        MatrixMatrixMultiply(proxy, nullptr, nullptr, n_gms, size, size, size);
        MatrixMatrixMultiply(proxy, nullptr, nullptr, n_gms, size, size, size);
        return nullptr;
    }
    return nullptr;
}

uint64_t* RknIteration(Party* proxy, uint64_t* x, uint64_t* z, uint64_t* ct1, uint32_t n_dim, uint32_t n_anc, uint32_t k_mer, uint64_t lambda, uint64_t alpha) {
    /* This function performs a single time point iteration of ppRKN inference.
     *
     * Input(s)
     * proxy: Party instance
     * x: data vector of size n_dim at time t
     * z: anchor points of size (k_mer * n_anc * n_dim)
     * ct1: the mapping from time t-1 of size ((k_mer + 1) * n_anc) -- the first n_anc elements are 1
     * n_dim: the number of elements to represent characters in the sequence
     * n_anc: the number of anchor points
     * k_mer: the k value of k-mers
     * lambda: the downscaling factor to decrease the effect of the previous time point
     *
     * Return(s)
     * ckt: the output of the mapping of the sequence after time point t
     */

    int p_role = proxy->GetPRole();

    if(p_role == proxy1 || p_role == proxy2) {
        uint32_t size = k_mer * n_anc * n_dim;
        uint32_t size2 = k_mer * n_anc;
        uint64_t* rep_x = new uint64_t[size];

        for(int i = 0; i < k_mer; i++) {
            for(int j = 0; j < n_anc; j++) {
                for(int k = 0; k < n_dim; k++) {
                    rep_x[(i * n_anc * n_dim) + (j * n_dim) + k] = x[k];
                }
            }
        }

        // computation of b_{k}[t]
        uint64_t* dp = DotProduct(proxy, rep_x, z, size, n_dim);
        uint64_t tmp_minus_one = ConvertToUint64(-1);
        for(uint32_t i = 0; i < size2; i++) {
            uint64_t tmp = dp[i] - (((uint64_t) 1 << FRACTIONAL_BITS) * p_role);
            dp[i] = LocalMultiply(alpha, tmp);
        }

        uint64_t* res_b = Exp(proxy, dp, size2);

        // computation of c_{k-1}[t-1] * b_{k}[t]
        uint64_t* skt = Multiply(proxy, res_b, ct1, size2);
        // computation of c_{k-1}[t-1] * b_{k}[t]
        uint64_t* ckt = new uint64_t[size2];
        for(uint32_t i = 0; i < size2; i++) {
            ckt[i] = LocalMultiply(lambda, skt[i]) +
                     LocalMultiply(((uint64_t) 1 << FRACTIONAL_BITS) - lambda, ct1[i + n_anc]); // this provides more precision
        }

        delete[] dp;
        delete[] res_b;
        delete[] skt;
        delete[] rep_x;

        return ckt;
    }
    else if(p_role == helper) {
        // n_dim contains "size"
        // n_anc contains "size2"
        uint32_t size = n_dim;
        uint32_t size2 = n_anc;

        DotProduct(proxy, nullptr, nullptr, size, 0);
        Exp(proxy, nullptr, size2);
        Multiply(proxy, nullptr, nullptr, size2);

        return nullptr;
    }
    return nullptr;
}


#endif //CECILIA_RKN_H
