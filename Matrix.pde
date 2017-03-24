class Matrix {
  int rows;
  int cols;
  float[][] matrix;

  // --------------------------------------------------------------
  Matrix (int r, int c) {
    this.rows = r;
    this.cols = c;
    this.matrix = new float[this.rows][this.cols];
  }

  // --------------------------------------------------------------
  int getRows () {
    return this.rows;
  }

  int getCols () {
    return this.cols;
  }

  float get (int r, int c) {
    float v;
    v = this.matrix[r][c];
    return v;
  }

  void set (float v, int r, int c) {
    this.matrix[r][c] = v;
  }

  void setAll (float[][] v) {
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        this.matrix[r][c] = v[r][c];
      }
    }
  }

  float[][] getAll () {
    float[][] v = new float[this.rows][this.cols];
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        v[r][c] = this.matrix[r][c];
      }
    }
    return v;
  }

  // --------------------------------------------------------------
  Matrix duplicate () {
    Matrix m = new Matrix(this.rows, this.cols);
    m.setAll(this.getAll());
    return m;
  }

  void zeroes () {
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        this.matrix[r][c] = 0.0;
      }
    }
  }

  void ones () {
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        this.matrix[r][c] = 1.0;
      }
    }
  }

  void identity () {
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        this.matrix[r][c] = (r == c) ? 1.0 : 0.0;
      }
    }
  }

  Matrix transpose () {
    Matrix ret = new Matrix(this.cols, this.rows);
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        ret.matrix[c][r] = this.matrix[r][c];
      }
    }
    return ret;
  }

  void randomize (RandomMode rmode) {
    float rnd;
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        switch (rmode) {
        case CENTERED:
          rnd = random(-1.0, 1.0);
          break;
        default:
          rnd = random(1.0);
        }
        this.matrix[r][c] = rnd;
      }
    }
  }

  void normRows () {
    float max;
    for (int r = 0; r < this.rows; r++) {
      max = -1.0;
      for (int c = 0; c < this.cols; c++) {
        if (max < this.matrix[r][c]) {
          max = this.matrix[r][c];
        }
      }
      if (max == 0) {
        println("ERROR! Cannot normRows by 0");
        return;
      }
      for (int c = 0; c < this.cols; c++) {
        this.matrix[r][c] = this.matrix[r][c] / max;
      }
    }
  }

  void normCols () {
    float max;
    for (int c = 0; c < this.cols; c++) {
      max = -1.0;
      for (int r = 0; r < this.rows; r++) {
        if (max < this.matrix[r][c]) {
          max = this.matrix[r][c];
        }
      }
      if (max == 0) {
        println("ERROR! Cannot normCols by 0");
        return;
      }
      for (int r = 0; r < this.rows; r++) {
        this.matrix[r][c] = this.matrix[r][c] / max;
      }
    }
  }

  float averageRow (int row) {
    float sum;
    sum = 0.0;
    for (int c = 0; c < this.cols; c++) {
      sum += this.matrix[row][c];
    }
    return (sum / this.cols);
  }

  float averageCol (int col) {
    float sum;
    sum = 0.0;
    for (int r = 0; r < this.rows; r++) {
      sum += this.matrix[r][col];
    }
    return (sum / this.rows);
  }

  float average () {
    float sum;
    sum = 0.0;
    for (int r = 0; r < this.rows; r++) {
      sum += this.averageRow(r);
    }
    return (sum / this.rows);
    //could also do:
    //for (int c = 0; c < this.cols; c++) {
    //  sum += this.averageCol(c);
    //}
    //return (sum / this.cols);
  }

  void normaverageRows () {
    float av;
    for (int r = 0; r < this.rows; r++) {
      av = this.averageRow(r);
      if (av == 0) {
        println("ERROR! Cannot normaverageRows by 0");
        return;
      }
      for (int c = 0; c < this.cols; c++) {
        this.matrix[r][c] = this.matrix[r][c] / av;
      }
    }
  }

  void normaverageCols () {
    float av;
    for (int c = 0; c < this.cols; c++) {
      av = this.averageCol(c);
      if (av == 0) {
        println("ERROR! Cannot normaverageCols by 0");
        return;
      }
      for (int r = 0; r < this.rows; r++) {
        this.matrix[r][c] = this.matrix[r][c] / av;
      }
    }
  }

  void div (float scalar) {
    this.mult(1.0/scalar);
  }

  void mult (float scalar) {
    if (scalar == 0) {
      println("ERROR! Cannot mult by 0");
      return;
    }
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        this.matrix[r][c] = this.matrix[r][c] * scalar;
      }
    }
  }

  Matrix mult (Matrix other) {
    return this.dot( other);
  }

  Matrix dot (Matrix other) {
    if (this.cols != other.rows) {
      println("ERROR! Can only multiply matrices where the number of cols of the left one equals the number of rows in the right one.");
      return new Matrix(0, 0);
    }
    Matrix res = new Matrix(this.rows, other.cols);
    float sum;
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < other.cols; j++) {
        sum = 0.0;
        for (int k = 0; k < this.cols; k++) {
          sum += this.matrix[i][k] * other.matrix[k][j];
        }
        res.matrix[i][j] = sum;
      }
    }
    return res;
  }

  //Multiplies the values at same indices row,col of 2 (same size) matrices
  //modifies this matrix by getting the (same size) resulting matrix
  void scale (Matrix other) {
    if ((this.cols != other.cols) || (this.rows != other.rows)) {
      println("ERROR! Can only scalar-multiply matrices of same size.");
      return;
    }
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.matrix[i][j] = this.matrix[i][j] * other.matrix[i][j];
      }
    }
  }

  void add (Matrix other) {
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        this.matrix[r][c] += other.matrix[r][c];
      }
    }
  }  

  void sub (Matrix other) {
    for (int r = 0; r < this.rows; r++) {
      for (int c = 0; c < this.cols; c++) {
        this.matrix[r][c] -= other.matrix[r][c];
      }
    }
  }  

  // --------------------------------------------------------------
  void mprint () {
    println(this.rows + "x" + this.cols + " matrix");
    for (int r = 0; r < this.rows; r++) {
      print("[ ");
      for (int c = 0; c < this.cols; c++) {
        print(this.matrix[r][c] + " ");
      }
      println("]");
    }
    println();
  }
}