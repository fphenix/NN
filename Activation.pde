enum Activation {
  SIGMOID, LOGISTIC, SOFTSTEP, // f(z) = 1 / (1 + e^(-z)); f'(z) = f(z).(1 - f(z))
  TANH, // f(z) = tanh(z) = ((e^z - e^(-z)) / ((e^z + e^(-z)); f'(z) = 1 - f(z)^2
  // TANH2, // f(z) = 2*sigmoid(2z) - 1 = (2 / (1 + e^(-2z)) - 1, f'(z) = 1 - f(z)^2
  RELU, // f(z) = max(0, z) = {0 for z < 0 else z; f'(z) = {0 for z < 0 else 1
  SOFTPLUS, // f(z) = ln(1+e^z) ; f'(z) = sigmoid(z) = 0 / (1+e^(-z)) 
  IDENTITY, LINEAR, //  f(z) = z; f'(z) = 1
  THRESHOLDS, // f(z) = {-1 for z < 0 else 1; f'(z) = {0 for z != 0 else unknown
  BINARYSTEP, // f(z) = {0 for z < 0 else 1; f'(z) = {0 for z != 0 else unknown
  GAUSSIAN,  // f(z) = e^(-z^2); f'(z) = -2z.f(z)

  PIECEWISE, // f(z) = { 0 if x <= xmin, 1 if x >= xmax, else mx+b ; f'(z) = { m if xmin > x > xmax else 0
  ARCTAN, // f(z) = tan^-1(z); f'(z) = 1 / (z^2 + 1)
  SOFTSIGN, // f(z) = z / (1 + abs(z)), f'(z) = 1 / (1 + abs(z))^2
  SINE  // f(z) = sin(z) ; f'(z) = cos(z)
}