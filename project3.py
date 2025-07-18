from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import matplotlib.animation as animation

G = nx.barabasi_albert_graph(100, 3) # random network w/ 100 ppl and 5% friend chance

for node in G.nodes:
    G.nodes[node]['state'] = 'U'  # start unaware all around
    G.nodes[node]['belief_age'] = 0  

num_start = int(0.2 * G.number_of_nodes())
for start in np.random.choice(G.nodes, size=num_start, replace=False):
    G.nodes[start]['state'] = 'B'  # 3 ppl start rumor
    G.nodes[start]['belief_age'] = 0  

def sim_network(G, expose_prob = 0.9, resist_prob = 0.1, correct_prob = 0.05, stay_b_prob = 0.25):
	new_states = {}
	for node in G.nodes:
		state = G.nodes[node]['state']
		friends = list(G.neighbors(node))
		b_friends = [n for n in friends if G.nodes[n]['state'] == 'B']
		if state == "U":
			if len(b_friends) > 0:
				if random.random() < expose_prob:
					new_states[node] = "B"
		if state == "B":
			G.nodes[node]['belief_age'] += 1
			age = G.nodes[node]['belief_age']
			if age > 5:
				rand = random.random()
				if rand < resist_prob:
					new_states[node] = "R"
				elif rand < resist_prob + correct_prob:
					new_states[node] = "F"
				elif rand < resist_prob + correct_prob + stay_b_prob:
					new_states[node] = "B"
				else:
					new_states[node] = "B"
	for node, new_state in new_states.items():
		G.nodes[node]['state']= new_state

def state_proportion(G):
	total = G.number_of_nodes()
	counts = {'U':0, 'B':0, 'R':0, 'F':0}
	for node in G.nodes:
		state = G.nodes[node]['state']
		counts[state] += 1
	return{
		'U': counts['U']/total,
		'B': counts['B']/total,
		'R': counts['R']/total,
		'F': counts['F']/total,
	}

pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(8, 6))
def update(frame):
	ax.clear()
	sim_network(G)
	color_map= []
	for node in G.nodes:
		state = G.nodes[node]['state']
		if state == 'U':
			color_map.append('gray')
		elif state == 'B':
			color_map.append('red')
		elif state == 'R':
			color_map.append('green')
		elif state == 'F':
			color_map.append('blue')
	
	nx.draw(G, pos, node_color = color_map, with_labels = False, node_size = 80, ax = ax)
	ax.set_title(f"Day {frame + 1}")

ani = animation.FuncAnimation(fig, update, frames = 100, interval = 300, repeat = True)
plt.show()

G = nx.barabasi_albert_graph(100, 3) # random network w/ 100 ppl and 5% friend chance

for node in G.nodes:
    G.nodes[node]['state'] = 'U'  # start unaware all around
    G.nodes[node]['belief_age'] = 0  

num_start = int(0.2 * G.number_of_nodes())
for start in np.random.choice(G.nodes, size=num_start, replace=False):
    G.nodes[start]['state'] = 'B'  # 3 ppl start rumor
    G.nodes[start]['belief_age'] = 0  

days = 100
u_val, b_val, r_val, f_val = [], [], [], []

for x in range(days):
	sim_network(G)
	proportions = state_proportion(G)
	u_val.append(proportions['U'])
	b_val.append(proportions['B'])
	r_val.append(proportions['R'])
	f_val.append(proportions['F'])

plt.figure()
plt.plot(range(days), u_val, "gray", label="Unaware (U)")
plt.plot(range(days), b_val, "red", label="Believers (B)")
plt.plot(range(days), r_val, "green", label="Resistant (R)")
plt.plot(range(days), f_val, "blue", label="Fact-checked (F)")
plt.title("Rumor/Belief Spread Over Time (Network)")
plt.xlabel("Time (days)")
plt.ylabel("Fraction of Population")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()


# DEFINE CONSTANTS
expose_rate = 1 # exposure rate
resist_rate = 1/14 # resistance rate
correction_frequency = 0.01 # how often believers are actually fact-checked

# # FUNCTION TO MAKE EXPOSURE RATE DYNAMIC- NOT WORKING
# def influence_factor(t, B_vals):
#     index = int(t)
#     if index >= len(B_vals):
#         index = len(B_vals) - 1
#     return 1 + 2 * B_vals[index]

# FUNCTION TO RETURN DERIVATIVES AT T
def f(y, t):
    U, B, R, F = y
    # influence = influence_factor(t, B_vals)  
    dU = -expose_rate * U * B
    dB = expose_rate * U * B - resist_rate * B - correction_frequency * B
    dR = resist_rate * B
    dF = correction_frequency * B

    return [dU, dB, dR, dF]


# INITIAL VALUES OF EACH FUNCTION 
U_0 = 1
B_0 = 3.125/(10**6)
R_0 = 0
F_0 = 0
y_0 = [U_0,B_0,R_0,F_0]

t = np.linspace(start=1,stop=100,num=101)
# B_vals = []
y = odeint(f,y_0,t) 


U = y[:,0]
B = y[:,1]
R = y[:,2]
F = y[:,3]

# B_vals = B

plt.figure()
plt.plot(t,U,"r",label="Unaware")
plt.plot(t,B,'b',label="Believer")
plt.plot(t,R,'g',label="Resistant") 
plt.plot(t, F, "k", label="Fact-checked")
peak_o = t[np.argmax(B)]
plt.axvline(peak_o, color='gray', linestyle='--', label="Peak Spread")
plt.title("UBRF Model w odeint")
plt.xlabel("Time (days)")
plt.ylabel("Fraction of Population")
plt.legend()
plt.show()

u_cache = {}
b_cache = {}
r_cache = {}
f_cache = {}

def u(t):
	if t == 0:
		return 1

	if t in u_cache:
		value = u_cache[t]
		return value

	result = u(t-1) - (u(t-1)*b(t-1))

	u_cache[t] = result

	return result

def b(t):
	if t == 0:
		return 3.125/(10**6)

	if t in b_cache:
		value = b_cache[t]
		return value

	result = b(t-1) + (u(t-1)*b(t-1)) - ((1/14)*b(t-1)) - (correction_frequency * b(t-1))

	b_cache[t] = result

	return result

def r(t):
	if t == 0:
		return 0

	if t in r_cache:
		value = r_cache[t]
		return value

	result = r(t-1)+((1/14)*b(t-1))

	r_cache[t] = result

	return result

def f(t):
    if t == 0:
        return 0
    if t in f_cache:
        return f_cache[t]

    result = f(t-1) + correction_frequency * b(t-1)
    f_cache[t] = result
    return result

plt.figure()
u_list = []
b_list = []
r_list = []
f_list = []
time_list = list(range(0, 101))
for day in range(0,101):
	u_list.append(u(day))
	b_list.append(b(day))
	r_list.append(r(day))
	f_list.append(f(day))

plt.plot(time_list,u_list,"r",label="Unaware")
plt.plot(time_list,b_list,'b',label="Believer")
plt.plot(time_list,r_list,'g',label="Resistant")
plt.plot(time_list, f_list, "k", label="Fact-check")
peak_r = time_list[np.argmax(u_list)]
plt.axvline(peak_r, color='gray', linestyle='--', label="Peak Spread")
plt.ylim((0, 1)) 
plt.title("UBRF Model w/ Recursion")
plt.xlabel("Time (days)")
plt.ylabel("Fraction of Population")
plt.legend()
plt.show()

plt.figure()

# odeint
plt.plot(t, U, "r--", label="U (odeint)")
plt.plot(t, B, "b--", label="B (odeint)")
plt.plot(t, R, "g--", label="R (odeint)")
plt.plot(t, F, "k--", label="F (odeint)")  

# recurszion
plt.plot(time_list, u_list, "r", label="U (recursion)")
plt.plot(time_list, b_list, "b", label="B (recursion)")
plt.plot(time_list, r_list, "g", label="R (recursion)")
plt.plot(time_list, f_list, "k", label="F (recursion)")

plt.title("UBRF Model: odeint vs Recursion")
plt.xlabel("Time (days)")
plt.ylabel("Fraction of Population")
plt.legend()
plt.ylim(0, 1)
plt.grid(True)
plt.show()
