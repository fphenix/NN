// FredL: 

Matrix X;
Matrix Ytarget;

NeuralNet nn;

void setup () {
  size(1000, 800);
  background(0);
  noFill();
  stroke(255);
  int nbDataSets = 4;
  float err;

  Matrix MtmpX = new Matrix(nbDataSets, 3); 
  // inputA, inputB, bias=1  
  float[][] tmpX = { {0.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0} };
  MtmpX.setAll(tmpX);

  Matrix MtmpY = new Matrix(nbDataSets, 1); 
  float[][] tmpY = { {0.0}, {1.0}, {1.0}, {0.0} };
  MtmpY.setAll(tmpY);

  nn = new NeuralNet(3, 4, 1, nbDataSets); // in , hid, out, nbdataset
  nn.setInputs(MtmpX.getAll());
  nn.setTargets(MtmpY.getAll());
  nn.setLearningRate(0.1);
  nn.setActiveType(Activation.TANH);
  nn.setOutputActiveType(Activation.SOFTPLUS); //RELU
  nn.setRandomMode(RandomMode.BINARY);

  int nbpoints = 2000;
  float pointstep = 1.0 * width / nbpoints;
  for (int i = 0; i < nbpoints; i++) {
    nn.epoch();
    err = nn.getCost();
    point(i * pointstep, map(err, 0.0, 1.0, height-10, 10));
    if (err < 0.003) {
      break;
    }
  }
  nn.hprint();
  nn.display(1);
}

void draw() {
}