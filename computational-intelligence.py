import pandas as pd               # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np                # linear algebra
from numpy.random import seed
from numpy.random import rand
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras import losses, optimizers
from keras.utils import to_categorical

from matplotlib import pyplot as plt
#%matplotlib inline

class Particle:
    
    def __init__(self, weights, velocity, omega, c1, c2, 
                 lower_bound, upper_bound, particle_id):
        self.id = particle_id
        self.weights = weights
        self.best_weights = weights
        self.velocity = np.array(velocity)
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.best_valuation = 0
        self.history = []
        
    def update(self, best_particle):
        self.update_velocity(best_particle)
        self.check_velocity()
        self.update_particle()
        # self.check_weights()
        
        return self.weights
    
    def update_velocity(self, bp):
        # v = Wv + c1r1(pi - xi) + c2r2(g - xi)
        # v        - ubrzanje 
        # W        - omega 
        # c1 i c2  - unapred zadati parametri
        # r1 i r2  - slucajni brojevi iz (0,1) uniformne raspodele
        # pi       - najbolje resenje trenutne jedinke
        # g        - najbojle globalno resenje
        r1 = np.random.random()
        r2 = np.random.random()
        
        self.velocity = self.omega * self.velocity + self.c1*r1*(np.add(np.array(self.best_weights), (-1)*np.array(self.weights))) + self.c2*r2*(np.add(np.array(bp), (-1)*np.array(self.weights))) 
        
    def check_velocity(self):
        for i in range(len(self.velocity)):
            if(i%2 == 1):
                for j in range(len(self.velocity[i])): 
                    if(self.velocity[i][j] < self.lower_bound):
                        self.velocity[i][j] = self.lower_bound
                    
                    if(self.velocity[i][j] > self.upper_bound): 
                        self.velocity[i][j] = self.upper_bound
            else: 
                for j in range(len(self.velocity[i])):
                    for k in range(len(self.velocity[i][j])):
            
                        if(self.velocity[i][j][k] < self.lower_bound):
                            self.velocity[i][j][k] = self.lower_bound
                
                        if(self.velocity[i][j][k] > self.upper_bound): 
                            self.velocity[i][j][k] = self.upper_bound
        
        return True
    '''
    def check_weights(self):
        
        return True
    '''
    
    def update_particle(self):
        # xi = xi + vi
        self.weights = np.add(np.array(self.weights), self.velocity)
        
    def update_valuation(self, valuation):
        self.history.append(valuation)
        
        if(valuation > self.best_valuation):
            self.best_valuation = valuation
            self.best_weights = self.weights
            
    def get_weights(self):
        return self.weights

class PSO:
    def __init__(self, num_particles, num_iters, training_x, training_y, model, shapes, c1, c2, w, lower_bound, upper_bound):
        self.num_iters = num_iters
        self.training_x = training_x
        self.training_y = training_y
        self.model = model
        self.best_particle = self.make_velocity(shapes)
        self.best_evaluation = 0
        self.history = []
        self.particles = self.make_particles(num_particles, shapes, c1, c2, w, lower_bound, upper_bound)
        self.evaluate_particles()
        self.combo = 0
        
        
    def make_particles(self, size, shape, c1, c2, w, low, upp):
        particles = []
        
        velocity = self.make_velocity(shape)
        
        for i in range(size):
            weights = self.make_weights(shape, low, upp)
            particles.append(Particle(weights, velocity, w, c1, c2, low, upp, i))
            
        return particles
    
    def make_velocity(self, shapes):
        velocity = []
        
        for shape in shapes:
            velocity.append(np.zeros(shape))
        
        return velocity
                
        
    def make_weights(self, shapes, low, upp):
        weights = []
        
        for shape in shapes:
            if(len(shape) == 1):
                # wght da dobijemo vrednosti iz (0, upp-low)
                wght = rand(shape[0])*(upp-low)
                rec = np.full(shape, low)
                
                weights.append(np.add(wght, rec))
                
            else:
                wght = rand(shape[0], shape[1])*(upp-low)
                rec = np.full(shape, low)
                
                weights.append(np.add(wght, rec))
                
        return weights
        
        
    def evaluate_particles(self):
        
        for particle in self.particles:
            self.model.set_weights(particle.get_weights())
            train_loss, train_acc = self.model.evaluate(self.training_x, self.training_y, verbose = 0)
            
            particle.update_valuation(train_acc)
            
            if(train_acc > self.best_evaluation):
                self.best_particle = particle.get_weights()
                self.best_evaluation = train_acc
        
        self.history.append(self.best_evaluation)
        
    def isclose(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    
    def update_particles(self):
        
        for particle in self.particles:
            particle.update(self.best_particle)
            
    def get_particles(self):
        return self.particles
    
    def start_pso(self):
        
        for i in range(self.num_iters):
            partly_best_solution = self.best_evaluation
            self.update_particles()
            self.evaluate_particles()
            
            #print("Iteracija: {}\nPreciznost: {}\n".format(i+1, self.best_evaluation))
            
            if(self.isclose(partly_best_solution, self.best_evaluation, 0.001)):
                self.combo += 1
            else:
                self.combo = 0
            
        return self.best_particle


# MAIN - iris
data = datasets.load_iris()
#data = datasets.load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(data.data, 
                                                    data.target, 
                                                    test_size=0.33)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Before scaler:\n', x_train[0])

scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print('\nAfter scaler:\n', x_train[0])

# Pravljenje neuronske mreze sa 3 sloja - ulazni, skriveni i izlazni
model = Sequential()
model.add(Dense(units=100, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(units=y_train.shape[1], activation='sigmoid'))
model.summary()

shapes = [i.shape for i in model.get_weights()]

model.compile(optimizer='adam', 
              loss=losses.categorical_crossentropy, 
              metrics=['accuracy'])

best_test_acc = 0

print('Making PSO: ')
for i in range(3):
    pso = PSO(20, 10, 
            x_train, y_train, 
            model, 
            shapes, 
            0.5, 1.0, 0.3, -2, 2)
    best_particle = pso.start_pso()
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    if(best_test_acc < test_acc): 
        best_test_acc = test_acc
    
    print("Current PSO test accuracy:   " + str(test_acc) + "\n")

print()
print("Global best accuracy: " + str(best_test_acc))
