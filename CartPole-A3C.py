# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
# 
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf
import os

import gym, time, random

# import threading

import multiprocessing as mp

from keras.models import *
from keras.layers import *
from keras import backend as K

#log and visualization.
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
start = time.time()

a_time = []
a_reward = []

def log_reward( R ):
	a_time.append( time.time() - start )
	a_reward.append( R )

#-- constants
ENV = 'CartPole-v0'

RUN_TIME = 30
THREADS = 64
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP  = .15
EPS_STEPS = 75000

MIN_BATCH = 1
LEARNING_RATE = 5e-3

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

#---------
class Brain:
	# train_queue = [  mp.Queue() for i in range(5) ]	# s, a, r, s', s' terminal mask
	# lock_queue = Lock()

	def __init__(self):
		
		self.session = tf.Session()
		K.set_session(self.session)
		K.manual_variable_initialization(True)

		self.model = self._build_model()
		self.graph = self._build_graph(self.model)

		self.session.run(tf.global_variables_initializer())
		self.default_graph = tf.get_default_graph()

		self.default_graph.finalize()	# avoid modifications

		# multiprocess global sample queue for batch traning.
		# self._train_queue = [ mp.Queue() for i in range(5) ]	# s, a, r, s', s' terminal mask
		self._train_queue = mp.Queue()
		self._train_lock = mp.Lock()

		# multiprocess global state queue for action predict
		self._predict_queue = mp.Queue()
		self._predict_lock = mp.Lock()

	def _build_model(self):

		l_input = Input( batch_shape=(None, NUM_STATE) )
		l_dense = Dense(16, activation='relu')(l_input)

		out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
		out_value   = Dense(1, activation='linear')(l_dense)

		model = Model(inputs=[l_input], outputs=[out_actions, out_value])
		model._make_predict_function()	# have to initialize before threading

		return model

	def _build_graph(self, model):
		s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
		a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
		r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward
		
		p, v = model(s_t)

		log_prob = tf.log( tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
		# log_prob = tf.reduce_sum( tf.log(p) * a_t, axis=1, keep_dims=True) + 1e-10
		advantage = r_t - v

		loss_policy = - log_prob * tf.stop_gradient(advantage)									# maximize policy
		loss_value  = LOSS_V * tf.square(advantage)												# minimize value error
		entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)	# maximize entropy (regularization)

		loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

		optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
		minimize = optimizer.minimize(loss_total)

		return s_t, a_t, r_t, minimize

	def batch_train(self):

		if self._train_queue.qsize() < MIN_BATCH:
			time.sleep(0)	# yield
			return

		if self._train_queue.qsize() < MIN_BATCH:	# more thread could have passed without lock
			return 									# we can't yield inside lock

		# s = np.array([])
		# a = np.array([])
		# r = np.array([])
		# s_ = np.array([])
		# s_mask = np.array([])

		# size = 1
		# while not self._train_queue.empty():
		# 	# print( self._train_queue[0].get() )
		# 	s = np.concatenate( (s, self._train_queue[0].get()) )
		# 	a = np.concatenate( (a, self._train_queue[1].get()) )
		# 	r = np.concatenate( (r, self._train_queue[2].get()) )
		# 	s_ = np.concatenate( (s_, self._train_queue[3].get()) )
		# 	s_mask_ = np.concatenate( (s_mask, self._train_queue[4].get()) )
		# 	size += 1
		# s, a, r, s_, s_mask = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
		with self._train_lock:

			if self._train_queue.empty():
				return

			q = self._train_queue
			while not q.empty():
				# print(q.get())
				q.get()
			return

			i = 0
			while not self._train_queue.empty():
				s_, a_, r_, s__, s_mask_ = self._train_queue.get()
				# if s__ == NONE_STATE:
				# 	shoooot = 1
				if i==0:
					s, a, r, s_, s_mask = s_, a_, r_, s__, s_mask_
				else:
					s = np.row_stack((s, s_))
					a = np.row_stack((a, a_))
					r = np.row_stack((r, r_))
					s_ = np.row_stack((s_, s__))
					s_mask = np.row_stack( (s_mask, s_mask_) )
				i += 1

				# print( s, a, r, s_, s_mask )

		# if len(s)==0:
		# 	return

		if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing train batch of %d" % len(s))

		v = self.predict_v(s_)
		r = r + GAMMA_N * v * s_mask	# set v to 0 where s_ is terminal state
		
		s_t, a_t, r_t, minimize = self.graph
		self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

	def batch_predict(self):
		global envs

		if self._predict_queue.qsize() < MIN_BATCH:
			time.sleep(0)	# yield
			return

			if self._predict_queue.qsize() < MIN_BATCH:	# more thread could have passed without lock
				return
				 									# we can't yield inside lock
			# print( self._predict_queue.qsize() )
			# print(self._predict_queue.empty())
		with self._predict_lock:
			# ids = []
			# s = []

			if self._predict_queue.empty():
				return

			i = 0
			id = []
			while not self._predict_queue.empty():
				id_, s_ = self._predict_queue.get()
				if i==0:
					s = s_
				else:
				# item = self._predict_queue.get()
				# print( item )
				# 	id = np.row_stack((id, id_))
					s = np.row_stack((s, s_))
				id.append(id_)
				i += 1

		# if len(s)==0:
		# 	return

		p = self.predict_p(np.array(s))

		for j in range(i):
			if id[j] < len(envs):
				envs[id[j]].agent.wait_q.put(p[j])

	def predict(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return p, v

	def predict_p(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return p

	def predict_v(self, s):
		with self.default_graph.as_default():
			p, v = self.model.predict(s)		
			return v

	def run(self):
		#while time.time()-start<RUN_TIME:
		while True:
			self.batch_predict()
			self.batch_train()

#---------
frames = 0
class Agent:
	def __init__(self, id, eps_start, eps_end, eps_steps, predict_queue, predict_lock, train_queue, train_lock):
		self.id = id
		self.eps_start = eps_start
		self.eps_end   = eps_end
		self.eps_steps = eps_steps

		self.memory = []	# used for n_step return
		self.R = 0.

		# for predicted nn output dispatching
		self.wait_q = mp.Queue(maxsize=1)
		self._predict_queue = predict_queue
		self._predict_lock = predict_lock

		# for training
		self._train_queue = train_queue
		self._train_lock = train_lock

	def getEpsilon(self):
		if(frames >= self.eps_steps):
			return self.eps_end
		else:
			return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate

	def act(self, s):
		eps = self.getEpsilon()			
		global frames; frames = frames + 1

		s = np.array([s])
		# put the state in the prediction q
		with self._predict_lock:
			self._predict_queue.put((self.id, s))
		# wait for the prediction to come back
		p = self.wait_q.get()

		# a = np.argmax(p)
		a = np.random.choice(NUM_ACTIONS, p=p)
		if random.random() < eps:
			a = random.randint(0, NUM_ACTIONS-1)
		return a
	
	def train(self, s, a, r, s_):

		def train_push(s, a, r, s_):
			# s_next = s_ is None ? NONE_STATE : s_
			# s_mask = s_ is None ? 0. : 1.
			with self._train_lock:
				if s_ is None:
					s_next = NONE_STATE
					s_mask = 0.
				else:
					s_next = s_
					s_mask = 1.
				s = np.array([s])
				a = np.array([a])
				r = np.array([r])
				s_next = np.array([s_next])
				s_mask = np.array([s_mask])
				self._train_queue.put( (s, a, r, s_next, s_mask) )

			# with self._train_lock:
			# 	self._train_queue[0].put(s)
			# 	self._train_queue[1].put(a)
			# 	self._train_queue[2].put(r)

			# 	if s_ is None:
			# 		self._train_queue[3].put(NONE_STATE)
			# 		self._train_queue[4].put(0.)
			# 	else:	
			# 		self._train_queue[3].put(s_)
			# 		self._train_queue[4].put(1.)

		def get_sample(memory, n):
			s, a, _, _  = memory[0]
			_, _, _, s_ = memory[n-1]

			return s, a, self.R, s_

		a_cats = np.zeros(NUM_ACTIONS)	# turn action into one-hot representation
		a_cats[a] = 1 

		self.memory.append( (s, a_cats, r, s_) )

		self.R = ( self.R + r * GAMMA_N ) / GAMMA

		if s_ is None:
			while len(self.memory) > 0:
				n = len(self.memory)
				s, a, r, s_ = get_sample(self.memory, n)
				train_push(s, a, r, s_)

				self.R = ( self.R - self.memory[0][2] ) / GAMMA
				self.memory.pop(0)
			# if len(self.memory) > 0:
			# 	n = len(self.memory)
			# 	s, a, r, s_ = get_sample(self.memory, n)
			# 	train_push(s, a, r, s_)

			# 	self.R = ( self.R - self.memory[0][2] ) / GAMMA
			# 	self.memory.clear()

			self.R = 0
			# return

		if len(self.memory) >= N_STEP_RETURN:
			s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
			train_push(s, a, r, s_)

			self.R = self.R - self.memory[0][2]
			self.memory.pop(0)
	
	# possible edge case - if an episode ends in <N steps, the computation is incorrect
		
#---------
# class Environment(threading.Thread):
class Environment(mp.Process):
	stop_signal = False

	def __init__(self, id, predict_queue, predict_lock, train_queue, train_lock, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
		# threading.Thread.__init__(self)
		mp.Process.__init__(self)

		self.id = id
		self.render = render
		self.env = gym.make(ENV)
		self.agent = Agent(id, eps_start, eps_end, eps_steps, predict_queue, predict_lock, train_queue, train_lock)

	def runEpisode(self):
		s = self.env.reset()

		R = 0
		while True:         
			time.sleep(THREAD_DELAY) # yield 

			if self.render: self.env.render()

			a = self.agent.act(s)
			s_, r, done, info = self.env.step(a)

			if done: # terminal state
				s_ = None

			self.agent.train(s, a, r, s_)

			s = s_
			R += r

			if done: #or self.stop_signal:
				break

		log_reward( R )
		# print("Total R:", R)

	def run(self):
		while not self.stop_signal:
			self.runEpisode()

	def stop(self):
		self.stop_signal = True

#-- main
env = gym.make(ENV)
NUM_STATE = env.env.observation_space.shape[0]
NUM_ACTIONS = env.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()	# brain is global in A3C
#env_test = Environment(id=0, predict_queue=brain._predict_queue, train_queue=brain._train_queue, train_lock=brain._train_lock, render=True, eps_start=0., eps_end=0.)
envs = [Environment(id=i, predict_queue=brain._predict_queue, predict_lock=brain._predict_lock, train_queue=brain._train_queue, train_lock=brain._train_lock) for i in range(THREADS)]

for e in envs:
	e.start()

brain.run()

# time.sleep(RUN_TIME)

for e in envs:
	e.stop()
for e in envs:
	e.join()

# opts.stop()

print("Training finished")

#plot rewards
plt.plot( a_time, a_reward )
plt.show()

#env_test.run()