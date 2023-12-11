//
// Created by Mete Akgun on 03.07.20.
//

#ifndef PML_CONSTANT_H
#define PML_CONSTANT_H

#define L_BIT 64
#define LP 67
#define FRACTIONAL_BITS 20
#define RING_SIZE 0xffffffffffffffff  // ring size
#define N1_MASK 0x7fffffffffffffff
#define N1 0x8000000000000000
#define EVEN_MASK 0xfffffffffffffff7

#define MAX_MULTIPLY 16384

#define PRECISION 100
#define MAX_SAMPLE 0xfffff
#define MAX_SAMPLE_MASK 0x7ffff

#define MAX_MATRIX_SIZE 20000
#define BUFFER_SIZE 400000000

#define DEBUG_FLAG 0


// constants for InverseSqrt
#define ORTHOGONAL_MASK 0x3e000
#define MAX_DELTA 100
#define MIN_DELTA 100
#define MAX_SCALAR 0xfffff
#define MAX_A 0x3fffff

// constants for sockets
#define SOCKET_NUMBER 8

enum Role {
    proxy1, proxy2, helper
};

enum Operation {
    // Core
    coreVectorisedMostSignificantBit,coreMostSignificantBit,coreEnd,coreMultiply,coreVectorisedMultiply,coreMultiplex,
    coreVectorisedMultiplex,coreVectorisedModularConversion,coreModularConversion,coreVectorisedCompare,coreCompare,
    coreExp, coreVectorisedExp,coreDotProduct,coreVectorisedDotProduct,coreMatrixMatrixMultiply,
    coreVectorisedMatrixMatrixMultiply,coreMatrixVectorMultiply,coreVectorisedMatrixVectorMultiply, coreDivide,
    coreVectorisedDivide, coreNormalise,coreSort,
    // AUC
    aucMostSignificantBit,aucDivide,aucVectorisedRound,aucVectorisedDivide,aucRocNoTie,aucRocWithTie,aucPrCurve,
    // CNN
    cnnMax, cnnVectorisedMax, cnnArgMax, cnnRelu, cnnVectorisedRelu, cnnDerivativeRelu, cnnVectorisedDerivativeRelu,
    cnnConvolutionalLayer, cnnFullyConnectedLayer,
    // RKN
    rknEigenDecomposition, rknVectorisedEigenDecomposition, rknGaussianKernel, rknInverseSqrt, rknVectorisedInverseSqrt,
    rknIteration
};


#endif //PML_CONSTANT_H
