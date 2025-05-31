// tests go here; this will not be compiled when this package is used as an extension.
// tests go here; this will not be compiled when this package is used as an extension.
namespace Torch.test {

    function printMatrix(matrix: number[][], label: string) {
        console.log(`\n${label}:`);
        for (let row of matrix) {
            console.log(row.join(" "));
        }
    }

    function runTests() {
        console.log("🚀 Starting Torch-C++ Backend Tests");

        // Sample matrices
        let A = [
            [1, 2],
            [3, 4]
        ];
        let B = [
            [5, 6],
            [7, 8]
        ];

        // Test Matrix Multiplication
        let resultMatmul = Torch.backend.matmul(2, 2, 2, 2, A, B);
        printMatrix(resultMatmul, "Matmul Result");

        // Test Addition
        let resultAdd = Torch.backend.add(4, A, B);
        printMatrix(resultAdd, "Addition Result");

        // Test Subtraction
        let resultSub = Torch.backend.sub(4, A, B);
        printMatrix(resultSub, "Subtraction Result");

        // Test Dot Product
        let flatA = Torch.backend.flattenMatrix(A);
        let flatB = Torch.backend.flattenMatrix(B);
        let resultDot = Torch.backend.dot(flatA.length, flatA, flatB);
        console.log(`Dot Product Result: ${resultDot}`);

        console.log("✅ All tests completed!");
    }

    // Run the tests
    runTests();
}
