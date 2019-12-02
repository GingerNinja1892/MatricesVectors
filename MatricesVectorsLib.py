"""
TODO:
Matrix determinants and inverses
"""

from numpy import matrix as np_matrix
from math import sin, cos, acos, pi
from numbers import Real

class Matrix:
    """
    Datatype 'Matrix' which is based off of a 2D python list.

    To create an identity matrix, provide an existing matrix from one of the static methods or a 2D array as 'matrix', otherwise the 1x1 zero-matrix will be created
    """

    def __init__(self, value=None):

        if isinstance(value, (Matrix, Vector)):
            self.value = value.value

        elif isinstance(value, list) and isinstance(value[0], list) and all([len(item) == len(value[0]) for item in value[1:]]):
            self.value = value

        elif isinstance(value, list) and all([isinstance(comp, Real) for comp in value]):
            self.value = Matrix.create_custom(len(value), value).value

        elif isinstance(value, np_matrix):   #can convert from numpy matrices
            self.value = value.tolist()

        elif value:    #if value is invalid
            raise ValueError("'value' must be a list or numpy matrix to be interpreted as a matrix")

        else:
            self.value = Matrix.create_zero(1).value

    @staticmethod
    def create_custom(num_rows, elements):
        """
        Static method to create a custom matrix from a 1 dimensional python list.
        """

        num_columns = len(elements) / num_rows
        if num_columns % 1 == 0:
            num_columns = round(num_columns)
        else:
            raise ValueError("'elements' length must be divisible by 'num_rows' - there must be a consistent number of columns")

        matrix = Matrix.create_zero(num_rows, num_columns).value

        index = 0
        for row_num in range(num_rows):
            for column_num in range(num_columns):
                matrix[row_num][column_num] = elements[index]
                index += 1

        return Matrix(matrix)

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
            for dummy in range(num_columns):
                matrix[row_num].append(0)

        return Matrix(matrix)

    @staticmethod
    def create_identity(num_rows, num_columns=None):
        """
        Static method to create an identity matrix. If columns not provided, assume square matrix.
        """

        if not num_columns:
            num_columns = num_rows

        matrix = Matrix.create_zero(num_rows, num_columns).value

        for row_num in range(num_rows):
            for column_num in range(num_columns):
                if row_num == column_num:
                    matrix[row_num][column_num] = 1

        return Matrix(matrix)

    @staticmethod
    def create_rotation_2D(angle):
        """
        Static method to create a rotation matrix to rotate a 2D position vector around 'angle' radians anti-clockwise
        """

        return Matrix([
            [round(cos(angle), 10), -round(sin(angle), 10)],
            [round(sin(angle), 10), round(cos(angle), 10)]
        ])

    @staticmethod
    def create_rotation_3D(angle, axis):
        """
        Static method to create a rotation matrix to rotate a 3D position vector 'angle' radians around the 'axis' axis anti-clockwise
        """

        if axis == "x":
            matrix = [
                [1, 0,                     0],
                [0, round(cos(angle), 10), -round(sin(angle), 10)],
                [0, round(sin(angle), 10), round(cos(angle), 10)]
            ]

        elif axis == "y":
            matrix = [
                [round(cos(angle), 10),  0, round(sin(angle), 10)],
                [0,                      1, 0],
                [-round(sin(angle), 10), 0, round(cos(angle), 10)]
            ]

        elif axis == "z":
            matrix = [
                [round(cos(angle), 10), -round(sin(angle), 10), 0],
                [round(sin(angle), 10), round(cos(angle), 10),  0],
                [0,                     0,                      1]
            ]

        else:
            raise ValueError("'axis' value must be either 'x', 'y' or 'z'")

        return Matrix(matrix)

    def __list__(self):
        return self.value

    def format(self):
        return [str(row).replace(",", "") for row in self.value]

    def __str__(self):
        return "\n".join(self.format())

    def __repr__(self):
        return "Matrix({})".format("\n       ".join(self.format()))

    def __len__(self):
        """
        Get the number of rows in the matrix
        """

        return len(self.value)

    def rows(self):
        """
        Get the number of rows in the matrix
        """

        return len(self)

    def columns(self):
        """
        Get the number of columns in the matrix
        """

        return len(self.value[0])

    def dimensions(self):
        """
        Get the dimensions of a matrix
        """

        return self.rows(), self.columns()

    def __add__(self, other):
        """
        Addition: can add 2 matrices if they have equal dimensions or a number to a matrix.
        """


        if isinstance(other, Real):   #if a real number
            return Matrix([[column + other for column in row] for row in self.value])

        if isinstance(other, Matrix) and self.dimensions() == other.dimensions(): #if another matrix with equal dimensions
            return Matrix([[self.value[row_num][column_num] + other.value[row_num][column_num] for column_num in range(len(self.value[0]))] for row_num in range(len(self.value))])

        elif isinstance(other, Matrix):
            raise ValueError("Cannot add matrices with different dimensions")

        else:
            return NotImplemented

    def __sub__(self, other):
        """
        Subtraction: can subtract 2 matrices if they have equal dimensions or a number from a matrix.
        """

        if isinstance(other, Real):   #if a real number
            return Matrix([[column - other for column in row] for row in self.value])

        if isinstance(other, Matrix) and self.dimensions() == other.dimensions(): #if another matrix with equal dimensions
            return Matrix([[self.value[row_num][column_num] - other.value[row_num][column_num] for column_num in range(len(self.value[0]))] for row_num in range(len(self.value))])

        elif isinstance(other, Matrix):
            raise ValueError("Cannot subtract matrices with different dimensions")

        else:
            return NotImplemented

    def __mul__(self, other):
        """
        Multiplication: can multiply 2 matrices using matrix multiplication or a scalar by a matrix
        """

        if isinstance(other, Matrix):
            rows_self = len(self.value)
            cols_self = len(self.value[0])
            rows_other = len(other.value)
            cols_other = len(other.value[0])

            if cols_self != rows_other:
                raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second to be able to multiply them")

            C = Matrix.create_zero(rows_self, cols_other).value
            for i in range(rows_self):
                for j in range(cols_other):
                    for k in range(cols_self):
                        C[i][j] += self.value[i][k] * other.value[k][j]

            return Matrix(C)

        elif isinstance(other, Real):
            return Matrix([[column * other for column in row] for row in self.value])

        else:
            return NotImplemented

    def __pow__(self, power):
        """
        Exponentiation: can raise a matrix to integer powers
        """

        if isinstance(power, int) and power >= 0:
            if power == 0:
                return 1
            elif power == 1:
                return self
            else:
                new = self
                for dummy in range(power - 1):
                    new *= self
                return new
        else:
            raise ValueError("'power' must be an integer greater than or equal to 0")

class Vector(Matrix):
    """
    Datatype 'Vector' which is a special form of a 'Matrix' with only 1 column
    """

    def __init__(self, vector=None):

        super().__init__(vector)
        assert self.columns() == 1, "Vectors must have only 1 column"

    @staticmethod
    def create_zero(self, dimensions):
        return super().create_zero(dimensions, 1)

    def point(self):
        """
        Converts the Vector into a point (a single dimensional list of components)
        """

        return [component[0] for component in self.value]

    def __repr__(self):
        return "Vector({})".format("\n       ".join(self.format()))

    def dimensions(self):
        """
        Get the dimension of the vector (only has 1 as you know the number of columns is 1)
        """

        return len(self)

    def __abs__(self):
        """
        Magnitude: Finds the magnitude of the vector
        """

        return sum([component[0] ** 2 for component in self.value]) ** 0.5

    def magnitude(self):
        """
        Finds the magnitude of the vector
        """

        return abs(self)

    def scalar_product(self, other):
        """
        Finds the scalar (dot) product of the 2 vectors
        """

        if not isinstance(other, Vector):
            other = Vector(other)

        assert len(self) == len(other), "Vectors must have the same dimensions"

        return sum([self.value[comp][0] * other.value[comp][0] for comp in range(len(self.value))])

    def angle(self, other):
        """
        Finds the angle between 2 vectors in radians
        """

        if not isinstance(other, Vector):
            other = Vector(other)

        return acos(round(abs(self.scalar_product(other) / (self.magnitude() * other.magnitude())), 10))

    def parallel(self, other):
        """
        Determines whether or not 2 vectors are parallel
        """

        return self.angle(other) % pi == 0

    def perpendicular(self, other):
        """
        Determines whether or not 2 vectors are perpendicular
        """

        return self.angle(other) % pi == pi / 2

class VectorLine:
    """
    Datatype 'VectorLine' which is a line described by a position vector and a direction vector. Doesn't inherit because it doesn't have a value anymore but a position and direction

    Direction is vector-like which represents the gradient of the line
    Position is vector-like which represents any point on the line. Default: The origin
    """

    def __init__(self, direction, position=None):

        self.dir = Vector(direction)
        if position is None:
            position = [0 for i in range(len(self.dir))]
        self.pos = Vector(position)
        assert len(self.pos) == len(self.dir), "Position and direction vectors must have the same dimensions"

    @staticmethod
    def line_between_points(A, B):
        """
        Returns a VectorLine which is the line between points A and B which are vector/point-like objects
        """

        A = Vector(B)
        B = Vector(B)

        assert len(A) == len(B), "must have the same dimensions"

        return VectorLine(Vector(Matrix.create_custom(len(A), [B.value[dimension][0] - A.value[dimension][0] for dimension in range(len(A))])), A)

    def format(self):
        return ["{}    {}".format(self.pos.value[index], self.dir.value[index]) for index in range(len(self)-1)], "{} + n{}".format(self.pos.value[-1], self.dir.value[-1])

    def __str__(self):
        out1, out2 = self.format()
        return "\n".join(out1) + "\n" + out2

    def __repr__(self):
        out1, out2 = self.format()
        return "VectorLine({})".format("\n           ".join(out1) + "\n           " + out2)

    def __len__(self):
        return len(self.dir)

    def rows(self):
        return len(self)

    def dimensions(self):
        return len(self)

    def is_point_on_line(self, point):
        """
        Determines whether or not a point is on this VectorLine
        """

        point = Vector(point)
        assert len(self) == len(point), "Point and line must have the same dimensions"

        lambdas = [(point.value[dimension][0] - self.pos.value[dimension][0]) / self.dir.value[dimension][0] for dimension in range(len(point))]
        if all([item == lambdas[0] for item in lambdas[1:]]):
            return True
        else:
            return False

    def intersection(self, other):
        """
        Determines whether or not 2 VectorLines intersect and if they do, where.
        """

        assert isinstance(other, VectorLine), "'other' must be a VectorLine"
        assert len(self) == len(other), "the lines must have the same dimensions"

        if self.pos.value == other.pos.value:
            return True, self.pos.point()
        elif self.dir.value == other.dir.value:
            return False, None

        lambda2 = ((self.pos.value[1][0] * self.dir.value[0][0]) + (self.dir.value[1][0] * other.pos.value[0][0]) - (self.pos.value[0][0] * self.dir.value[1][0]) - (other.pos.value[1][0] * self.dir.value[0][0])) / ((other.dir.value[1][0] * self.dir.value[0][0]) - (other.dir.value[0][0] * self.dir.value[1][0]))
        lambda1 = (other.pos.value[0][0] - self.pos.value[0][0] + (lambda2 * other.dir.value[0][0])) / self.dir.value[0][0]

        point = [
            other.pos.value[0][0] + (lambda2 * other.dir.value[0][0]),  #x
            other.pos.value[1][0] + (lambda2 * other.dir.value[1][0])   #y
        ]

        intersect = True
        i = 2

        while i < len(self.pos.value) and intersect:     #will for each extra dimension after x and y

            e1 = self.pos.value[i][0] + (lambda1 * self.dir.value[i][0])      #extra dimension (whatever that is) for line 1
            e2 = other.pos.value[i][0] + (lambda2 * other.dir.value[i][0])      #extra dimension (whatever that is) for line 2
            if e1 == e2:
                point.append(e1)                #if they are equal then they intersect in all dimensions up to and including this dimension so we add the point in this dimensions in which they intersect
            else:
                intersect = False

            i += 1

        return intersect, point

    def status(self, other):
        """
        Determines the status of 2 VectorLines: they could be parallel, identical, skew, intersect, intersect perpendicular or skew perpendicular
        """

        assert isinstance(other, VectorLine), "'other' must be a VectorLine"
        assert len(self) == len(other), "the lines must have the same dimensions"

        point = None
        angle = self.dir.angle(other.dir) % pi

        if angle == 0:      #if the angle between them is 0, they are either parallel or the same line
            if self.is_point_on_line(other.pos):                                #if the position vector of 1 line is on the other line and they are parallel, they are the same line
                status = "identical"
            else:                                                               #otherwise they are parallel
                status = "parallel"

        else:               #they are neither parallel nor perpendicular but can either intersect or be skew
            intersect, point = self.intersection(other)
            if intersect:
                status = "intersect"
            else:
                status = "skew"

            if angle == pi/2:   #if the angle between them is 90, they are perpendicular and either skew or they intersect
                status += " perpendicular"

        return status, point

class VectorPlane:

    def __init__(self, direction_vector1, direction_vector2, position_vector=None):

        assert isinstance(direction_vector1, Vector) and not all([comp == 0 for comp in direction_vector1])
        assert isinstance(direction_vector2, Vector) and not all([comp == 0 for comp in direction_vector2])
        assert len(direction_vector1) == len(direction_vector2)
        assert not direction_vector1.parallel(direction_vector2)

        if position_vector is None:
            position_vector = Vector.create_zero(len(direction_vector1))

        self.position_vector = position_vector
        self.direction_vector1 = direction_vector1
        self.direction_vector2 = direction_vector2
