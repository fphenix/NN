class NeuralNet {
  int nbDataSets;      // number of input data sets
  int inputLayerSize;  // number of inputs pad (hours of sleep, hours studying)
  int outputLayerSize; // number of output pads (grade 0 to 100)
  int hiddenLayerSize; // number of hidden nodes/neurons
  boolean showhidden = false;
  Activation activationType;
  Activation outputActivationType;
  RandomMode randomMode;

  Matrix X;
  Matrix XT;
  Matrix W1;
  Matrix dJdW1;
  Matrix Z2;
  Matrix A2;
  Matrix A2T;
  Matrix W2;
  Matrix W2T;
  Matrix dJdW2;
  Matrix Z3;
  Matrix delta3; // backpropagating error
  Matrix delta2;
  Matrix Yhat;
  Matrix Ytarget;
  Matrix Error; // error cost
  float Jcost;
  float learningRate;

  // --------------------------------------------------------------
  NeuralNet (int tin, int thid, int tout, int tds) {
    this.nbDataSets      = tds;
    this.inputLayerSize  = tin;
    this.hiddenLayerSize = thid;
    this.outputLayerSize = tout;
    this.learningRate = 0.1;
    this.activationType = Activation.SIGMOID;
    this.outputActivationType = Activation.SIGMOID;
    this.randomMode = RandomMode.BINARY;
    this.Jcost = 1.0;
    this.init();
  }

  // --------------------------------------------------------------
  void init () {
    this.W1   = new Matrix(this.inputLayerSize, this.hiddenLayerSize);
    this.Z2   = new Matrix(this.nbDataSets, this.hiddenLayerSize);
    this.A2   = new Matrix(this.nbDataSets, this.hiddenLayerSize);
    this.A2T  = new Matrix(this.hiddenLayerSize, this.nbDataSets);
    this.W2   = new Matrix(this.hiddenLayerSize, this.outputLayerSize);
    this.W2T  = new Matrix(this.outputLayerSize, this.hiddenLayerSize);
    this.Z3   = new Matrix(this.nbDataSets, this.outputLayerSize);
    this.Yhat = new Matrix(this.nbDataSets, this.outputLayerSize);
    this.Ytarget = new Matrix(this.nbDataSets, this.outputLayerSize);
    this.Error = new Matrix(this.nbDataSets, this.outputLayerSize);
    this.delta3 = new Matrix(this.nbDataSets, this.outputLayerSize);
    this.delta2 = new Matrix(this.nbDataSets, this.hiddenLayerSize);
    this.dJdW2 = new Matrix(this.hiddenLayerSize, this.outputLayerSize);
    this.dJdW1 = new Matrix(this.inputLayerSize, this.hiddenLayerSize);

    this.W1.randomize(this.randomMode);
    this.W2.randomize(this.randomMode);
    this.W2T = this.W2.transpose();
  }

  // --------------------------------------------------------------
  void setActiveType (Activation type) {
    this.activationType = type;
  }

  void setRandomMode (RandomMode mode) {
    this.randomMode = mode;
  }

  void setOutputActiveType (Activation type) {
    this.outputActivationType = type;
  }

  void setInputs (float[][] xin) {
    this.X = new Matrix (this.nbDataSets, this.inputLayerSize);
    this.X.setAll(xin);
    this.XT = this.X.transpose();
  }

  void setLearningRate (float lr) {
    this.learningRate = lr;
  }

  void setTargets (float[][] targ) {
    this.Ytarget.setAll(targ);
  }

  float getCost () {
    return this.Jcost;
  }

  //--------------------------------------------------------------
  // 0 to 1
  float sigmoidCalc (float x) {
    return (1.0 / (1.0 + exp(-x)));
  }

  float sigmoidPrimeCalc (float x) {
    float s = this.sigmoidCalc(x);
    return (s * (1.0 - s));
  }

  //--------------------------------------------------------------
  // -1 to +1
  float tanhCalc (float x) {
    float ex = exp(x);
    float emx = exp(-x);
    return ((ex - emx) / (ex + emx));
  }

  float tanhPrimeCalc (float x) {
    float t = this.tanhCalc(x);
    return (1.0 - (t * t));
  }

  //--------------------------------------------------------------
  // 0 to +inf ; ReLU rectified linear unit
  float reluCalc (float x) {
    return max(0, x);
  }

  float reluPrimeCalc (float x) {
    return (x < 0) ? 0.0 : 1.0;
  }

  //--------------------------------------------------------------
  // f(z) = {-1 for z < 0 else 1; f'(z) = {0 for z != 0 else unknown
  float thresholdCalc (float x) {
    return (x <= 0) ? -1.0 : 1.0;
  }

  float thresholdPrimeCalc (float x) {
    return (x == 0) ? 1.0 : 0.0;
  }

  //--------------------------------------------------------------
  // f(z) = {0 for z < 0 else 1; f'(z) = {0 for z != 0 else unknown
  float binarystepCalc (float x) {
    return (x <= 0) ? 0.0 : 1.0;
  }

  float binarystepPrimeCalc (float x) {
    return (x == 0) ? 1.0 : 0.0;
  }

  //--------------------------------------------------------------
  // 0 to +inf ; SOFTPLUS: smoother ReLU
  float softplusCalc (float x) {
    return log(1.0 + exp(x));
  }

  float softplusPrimeCalc (float x) {
    return this.sigmoidCalc(x);
  }

  //--------------------------------------------------------------
  // -inf to +inf ; Identity/Linear
  float identityCalc (float x) {
    return x;
  }

  float identityPrimeCalc (float x) {
    return 1.0 + (0.0 * x); // just 1 : not useful for backpropagation!!!!
  }

  //--------------------------------------------------------------
  // f(z) = e^(-z^2); f'(z) = -2z.f(z)
  float gaussianCalc (float x) {
    return exp(-(x*x));
  }

  float gaussianPrimeCalc (float x) {
    return -2.0 * x * this.gaussianCalc(x);
  }

  // --------------------------------------------------------------
  Matrix activation (Matrix mtx, Activation type) {
    float rv, mv;
    Matrix ret = new Matrix(mtx.rows, mtx.cols);
    for (int r = 0; r < mtx.rows; r++) {
      for (int c = 0; c < mtx.cols; c++) {
        mv = mtx.matrix[r][c];
        switch (type) {
        case TANH:
          rv = this.tanhCalc(mv);
          break;
        case RELU:
          rv = this.reluCalc(mv);
          break;
        case SOFTPLUS:
          rv = this.softplusCalc(mv);
          break;
        case IDENTITY:
        case LINEAR:
          rv = this.identityCalc(mv);
          break;
        case GAUSSIAN:
          rv = this.gaussianCalc(mv);
          break;
        case THRESHOLDS:
          rv = this.thresholdCalc(mv);
          break;
        case BINARYSTEP:
          rv = this.binarystepCalc(mv);
          break;
        default:
          rv = this.sigmoidCalc(mv);
          break;
        }
        ret.matrix[r][c] = rv;
      }
    }
    return ret;
  }

  Matrix activationPrime (Matrix mtx, Activation type) {
    float rv, mv;
    Matrix ret = new Matrix(mtx.rows, mtx.cols);
    for (int r = 0; r < mtx.rows; r++) {
      for (int c = 0; c < mtx.cols; c++) {
        mv = mtx.matrix[r][c];
        switch (type) {
        case TANH:
          rv = this.tanhPrimeCalc(mv);
          break;
        case RELU:
          rv = this.reluPrimeCalc(mv);
          break;
        case SOFTPLUS:
          rv = this.softplusPrimeCalc(mv);
          break;
        case IDENTITY:
        case LINEAR:
          rv = this.identityPrimeCalc(mv); // not useful for backpropagation
          break;
        case GAUSSIAN:
          rv = this.gaussianPrimeCalc(mv);
          break;
        case THRESHOLDS:
          rv = this.binarystepPrimeCalc(mv);
          break;
        case BINARYSTEP:
          rv = this.thresholdPrimeCalc(mv);
          break;
        default:
          rv = this.sigmoidPrimeCalc(mv);
          break;
        }
        ret.matrix[r][c] = rv;
      }
    }
    return ret;
  }

  // --------------------------------------------------------------
  float calcCost () {
    float j = 0.0; // float[outputLayerSier]
    float tmp;
    for (int i = 0; i < this.nbDataSets; i++) {
      tmp = this.Error.get(i, 0);
      j += tmp * tmp;
    }
    return (0.5 * j);
  }

  void cost() {
    Matrix fprimeZ3 = new Matrix(this.nbDataSets, this.outputLayerSize);
    Matrix fprimeZ2 = new Matrix(this.nbDataSets, this.hiddenLayerSize);

    this.Error = this.Ytarget.duplicate();
    this.Error.sub(this.Yhat);

    this.Jcost = this.calcCost();

    fprimeZ3 = this.activationPrime(this.Z3, this.outputActivationType);
    fprimeZ2 = this.activationPrime(this.Z2, this.activationType);

    // d(3) = -(y - yh).f'(z(3)) ; error = (y - yh)
    this.delta3 = this.Error.duplicate();
    this.delta3.mult(-1);
    this.delta3.scale(fprimeZ3);
    // dJ/dW(2) = (a(2))T.d(3)
    this.dJdW2 = this.A2T.dot(this.delta3);

    // d(2) = d(3).(W(2))T.f'(z(2))
    this.delta2 = this.delta3.dot(this.W2T);
    this.delta2.scale(fprimeZ2);
    // dJ/dW(1) = (X)T.d(2)
    this.dJdW1 = this.XT.dot(this.delta2);
  }

  // --------------------------------------------------------------
  void epoch () {
    this.forward();
    this.cost();
    this.backward();
  }

  void forward () {
    this.Z2 = this.X.dot(this.W1);
    this.A2 = this.activation(this.Z2, this.activationType);
    this.A2T = this.A2.transpose();

    this.Z3 = this.A2.dot(this.W2);
    this.Yhat = this.activation(this.Z3, this.outputActivationType);
  }

  void backward () {
    this.dJdW1.mult(-1.0 * this.learningRate);
    this.W1.add(this.dJdW1);
    this.dJdW2.mult(-1.0 * this.learningRate);
    this.W2.add(this.dJdW2);
  }

  // --------------------------------------------------------------
  void bprint() {
    println("A2T: ");
    this.A2T.mprint();
    println("delta3: ");
    this.delta3.mprint();
    println("dJdW2: ");
    this.dJdW2.mprint();
    println("XT: ");
    this.XT.mprint();
    println("delta2: ");
    this.delta2.mprint();
    println("dJdW1: ");
    this.dJdW1.mprint();
  }

  void hprint() {
    this.showhidden = true;
    nprint();
  }

  void nprint () {
    println("X Inputs: ");
    this.X.mprint();
    if (this.showhidden) {
      println("W1 Weights (input): ");
      this.W1.mprint();
      println("Z2 Weigthed (input): ");
      this.Z2.mprint();
      println("A2 Activation (input): ");
      this.A2.mprint();
      println("W2 Weights (output): ");
      this.W2.mprint();
      println("Z3 Weigthed (output): ");
      this.Z3.mprint();
    }
    println("Yhat Outputs (Estimation): ");
    this.Yhat.mprint();
    println("Ytarg Outputs (Target): ");
    this.Ytarget.mprint();
    if (this.showhidden) {
      println("Error (target - estimation): ");
      this.Error.mprint();
      println("Error Cost: " + this.Jcost);
      println();
    }    
    this.showhidden = false;
  }

  void display (int datasetnum) {
    int insize = this.inputLayerSize;
    int hidsize = this.hiddenLayerSize;
    int w1size = insize*hidsize;
    int outsize = this.outputLayerSize;
    int w2size = hidsize*outsize;
    int xpad = 50;
    int ypad = 20;
    int xstep = 80;
    int ystep = 30;
    
    pushMatrix();
    textAlign(CENTER, CENTER);
    ellipseMode(CENTER);
    rectMode(CENTER);
    translate(xpad, height/2);

    //Inputs X ------------------------------------------------------
    pushMatrix();
    translate(0, (-insize/2.0)*ystep);
    text("Inputs", 0, -ystep);
    for (int i = 0; i < insize; i++) {    
      text(this.X.get(datasetnum, i), 0, ystep*i);
    }
    popMatrix();
    translate(xstep, 0);

    // W1 ------------------------------------------------------
    pushMatrix();
    translate(0, (-w1size/2.0)*ystep);
    text("W1", 0, -ystep);
    for (int i = 0; i < hidsize; i++) {    
      for (int j = 0; j < insize; j++) {    
        text(this.W1.get(j, i), 0, ystep*((i*insize)+j));
      }
    }
    popMatrix();
    translate(xstep, 0);

    // hidden nodes : Z2 and A2 --------------------------------
    pushMatrix();
    translate(0, (-hidsize/2.0)*ystep);
    text("Z2", 0, -ystep);
    text("A2", xstep, -ystep);
    for (int i = 0; i < hidsize; i++) {    
      text(this.Z2.get(datasetnum, i), 0, ystep*i);
      text(this.A2.get(datasetnum, i), xstep, ystep*i);
    }
    popMatrix();
    translate(xstep, 0);
    translate(xstep, 0);

    // W2 ------------------------------------------------------
    pushMatrix();
    translate(0, (-w2size/2.0)*ystep);
    text("W2", 0, -ystep);
    for (int i = 0; i < outsize; i++) {    
      for (int j = 0; j < hidsize; j++) {    
        text(this.W2.get(j, i), 0, ystep*((i*hidsize)+j));
      }
    }
    popMatrix();
    translate(xstep, 0);

    // Z3 and A3 = Yhat ----------------------------------------------
    pushMatrix();
    translate(0, (-outsize/2.0)*ystep);
    text("Z3", 0, -ystep);
    text("Yestimated", xstep, -ystep);
    for (int i = 0; i < outsize; i++) {    
      text(this.Z3.get(datasetnum, i), 0, ystep*i);
      text(this.Yhat.get(datasetnum, i), xstep, ystep*i);
    }
    popMatrix();
    translate(xstep, 0);
    translate(xstep, 0);

    // Ytarget and Error -----------------------------------------
    pushMatrix();
    translate(0, (-outsize/2.0)*ystep);
    text("Yexpected", 0, -ystep);
    text("Error", xstep, -ystep);
    for (int i = 0; i < outsize; i++) {    
      text(this.Ytarget.get(datasetnum, i), 0, ystep*i);
      text(this.Error.get(datasetnum, i), xstep, ystep*i);
    }
    popMatrix();
    translate(xstep, 0);
    translate(xstep, 0);

    popMatrix();
  }
}