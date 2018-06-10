from numpy import matrix as np_matrix

class Matrix:
    """
    Datatype 'Matrix' which is based off of a 2D python list.

    To create an identity matrix, provide either:
        an existing matrix from one of the static methods or a 2D array as 'matrix'
        'num_rows' and 'elements'
    Providing nothing will result in the zero matrix being created
    """

    def __init__(self, num_rows=None, elements=None, matrix=None):

        if isinstance(matrix, np_matrix):
            self.matrix = matrix.tolist()
        
        elif num_rows and elements:
            self.matrix = Matrix.create_custom(num_rows, elements)
        
        elif matrix:
            self.matrix = matrix
        
        else:
            self.matrix = Matrix.create_zero

    @staticmethod
    def create_custom(num_rows, elements):
        """
        Static method to create a custom matrix from a 1 dimensional python list.
        """

        num_columns = len(elements) // num_rows

        matrix = Matrix.create_zero(num_rows, num_columns)

        index = 0
        for row_num in range(num_rows):
            for column_num in range(num_columns):
                matrix[row_num][column_num] = elements[index]
                index += 1
        
        return matrix
    
    @staticmethod
    def create_zero(num_rows, num_columns=None):
        """
        Static method to create a zero matrix. If columns not provided, assume square matrix.
        """

        if not num_columns:
            num_columns = num_rows

        matrix = []
        for row_num in range(num_rows):
            matrix.append([])
            for column_num in range(num_columns):
                matrix[row_num].append(0)
        
        return matrix
    
    @staticmethod
    def create_identity(num_rows, num_columns=None):
        """
        Static method to create an identity matrix. If columns not provided, assume square matrix.
        """

        if not num_columns:
            num_columns = num_rows

        matrix = Matrix.create_zero(rows, columns)

        for row_num in range(num_rows):
            for column_num in range(num_columns):
                if row_num == column_num:
                    matrix[row_num][column_num] = 1
        
        return matrix