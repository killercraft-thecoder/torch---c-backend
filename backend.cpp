namespace Torch {
    namespace backend {

        // Matrix multiplication (safe & optimized)
        void matmul(int rowsA, int colsA, int rowsB, int colsB, float* A, float* B, float* result) {
            if (colsA != rowsB) return; // Dimension mismatch

            for (int r = 0; r < rowsA; r++) {
                for (int c = 0; c < colsB; c++) {
                    result[r * colsB + c] = 0;
                    for (int i = 0; i < colsA; i++) {
                        result[r * colsB + c] += A[r * colsA + i] * B[i * colsB + c];
                    }
                }
            }
        }

        // Tensor addition
        void add(int size, float* A, float* B, float* result) {
            for (int i = 0; i < size; i++) {
                result[i] = A[i] + B[i];
            }
        }

        // Tensor subtraction
        void sub(int size, float* A, float* B, float* result) {
            for (int i = 0; i < size; i++) {
                result[i] = A[i] - B[i];
            }
        }

        // Memory-safe initialization of a tensor
        void init_tensor(int rows, int cols, float* tensor) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    tensor[r * cols + c] = 0.0f; // Ensures all values are initialized safely
                }
            }
        }

        // Element-wise multiplication
        void mul(int size, float* A, float* B, float* result) {
            for (int i = 0; i < size; i++) {
                result[i] = A[i] * B[i];
            }
        }

        // Tensor dot product (reduces large computations)
        float dot(int size, float* A, float* B) {
            float sum = 0.0f;
            for (int i = 0; i < size; i++) {
                sum += A[i] * B[i];
            }
            return sum;
        }
    }
}
