from MatricesVectorsLib import Matrix

class create_zero():

    @staticmethod
    def whenCreateZero_allValuesZero():
        matrix = Matrix.create_zero(2,2)
        for row in matrix:
            for column in row:
                assert column == 0
    
    @staticmethod
    def whenCreateZero_infersNumColumns():
        matrix = Matrix.create_zero(2)
        assert len(matrix[0]) == 2

class init():

    @staticmethod
    def whenCreateInstanceFromNumpy_getsCorrectValues():
        from numpy import matrix as np_matrix
        values = [[1,2,3],[4,5,6]]
        matrix = np_matrix(values)
        matrix = Matrix(matrix=matrix)
        assert matrix.matrix == values

if __name__ == "__main__":
    create_zero.whenCreateZero_allValuesZero()
    create_zero.whenCreateZero_infersNumColumns()
    init.whenCreateInstanceFromNumpy_getsCorrectValues()
    print("All tests OK")