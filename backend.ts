namespace Torch.backend {
    
    // Utility function: Convert 2D matrix to 1D array (row-major order)
    function flattenMatrix(matrix: number[][]): number[] {
        let flatArray: number[] = [];
        for (let row of matrix) {
            flatArray = flatArray.concat(row);
        }
        return flatArray;
    }

    // Utility function: Convert 1D array back to 2D matrix
    function unflattenMatrix(flatArray: number[], rows: number, cols: number): number[][] {
        let matrix: number[][] = [];
        for (let r = 0; r < rows; r++) {
            matrix[r] = flatArray.slice(r * cols, (r + 1) * cols);
        }
        return matrix;
    }

    //% shim=Torch.backend::__matmul
    export function matmul(rowsA: number, colsA: number, rowsB: number, colsB: number, A: number[][], B: number[][]): number[][] {
        let flatA = flattenMatrix(A);
        let flatB = flattenMatrix(B);
        let resultFlat = __matmul(rowsA, colsA, rowsB, colsB, flatA, flatB);
        return unflattenMatrix(resultFlat, rowsA, colsB);
    }

    //% shim=Torch.backend::__add
    export function add(size: number, A: number[][], B: number[][]): number[][] {
        let flatA = flattenMatrix(A);
        let flatB = flattenMatrix(B);
        let resultFlat = __add(size, flatA, flatB);
        return unflattenMatrix(resultFlat, A.length, A[0].length);
    }

    //% shim=Torch.backend::__sub
    export function sub(size: number, A: number[][], B: number[][]): number[][] {
        let flatA = flattenMatrix(A);
        let flatB = flattenMatrix(B);
        let resultFlat = __sub(size, flatA, flatB);
        return unflattenMatrix(resultFlat, A.length, A[0].length);
    }

    //% shim=Torch.backend::__init_tensor
    export function initTensor(rows: number, cols: number): number[][] {
        let resultFlat = __init_tensor(rows, cols);
        return unflattenMatrix(resultFlat, rows, cols);
    }

    //% shim=Torch.backend::__mul
    export function mul(size: number, A: number[][], B: number[][]): number[][] {
        let flatA = flattenMatrix(A);
        let flatB = flattenMatrix(B);
        let resultFlat = __mul(size, flatA, flatB);
        return unflattenMatrix(resultFlat, A.length, A[0].length);
    }

    //% shim=Torch.backend::__dot
    export function dot(size: number, A: number[][], B: number[][]): number {
        let flatA = flattenMatrix(A);
        let flatB = flattenMatrix(B);
        return __dot(size, flatA, flatB);
    }
}
