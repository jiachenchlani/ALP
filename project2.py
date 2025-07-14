from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# DEFINE CONSTANTS
a = 1 # infection rate
b = 1/14 # recovery rate
d_r = 0.01 

# FUNCTION TO RETURN DERIVATIVES AT T
def f(y,t):
	S, I, R, D = y # get previous values of S, I, R and store them in array y
	d0 = -a*S*I # derivative of S(t)
	d1 = a*S*I - b*I - d_r*I # derivative of I(t)
	d2 = b*I # derivative of R(t)
	d3 = d_r*I

	return [d0, d1, d2, d3]

# INITIAL VALUES OF EACH FUNCTION 
S_0 = 1
I_0 = 3.125/(10**6)
R_0 = 0
D_0 = 0
y_0 = [S_0,I_0,R_0,D_0]

t = np.linspace(start=1,stop=100,num=101)
y = odeint(f,y_0,t) 

S = y[:,0]
I = y[:,1]
R = y[:,2]
D = y[:,3]

plt.figure()
plt.plot(t,S,"r",label="S(t)")
plt.plot(t,I,'b',label="I(t)")
plt.plot(t,R,'g',label="R(t)")
plt.plot(t, D, "k", label="D(t)")
peak_o = t[np.argmax(I)]
plt.axvline(peak_o, color='gray', linestyle='--', label="Peak Infections")
plt.title("SIRD Model w odeint")
plt.xlabel("Time (days)")
plt.ylabel("Fraction of Population")
plt.legend()
plt.show()

s_cache = {}
i_cache = {}
r_cache = {}
d_cache = {}

def s(t):
	if t == 0:
		return 1

	if t in s_cache:
		value = s_cache[t]
		return value

	result = s(t-1) - (s(t-1)*i(t-1))

	s_cache[t] = result

	return result

def i(t):
	if t == 0:
		return 3.125/(10**6)

	if t in i_cache:
		value = i_cache[t]
		return value

	result = i(t-1) + (s(t-1)*i(t-1)) - ((1/14)*i(t-1)) - (d_r * i(t-1))

	i_cache[t] = result

	return result

def r(t):
	if t == 0:
		return 0

	if t in r_cache:
		value = r_cache[t]
		return value

	result = r(t-1)+((1/14)*i(t-1))

	r_cache[t] = result

	return result

def d(t):
    if t == 0:
        return 0
    if t in d_cache:
        return d_cache[t]

    result = d(t-1) + d_r * i(t-1)
    d_cache[t] = result
    return result

plt.figure()
s_list = []
i_list = []
r_list = []
d_list = []
time_list = list(range(0, 101))
for day in range(0,101):
	s_list.append(s(day))
	i_list.append(i(day))
	r_list.append(r(day))
	d_list.append(d(day))

plt.plot(time_list,s_list,"r",label="s(t)")
plt.plot(time_list,i_list,'b',label="i(t)")
plt.plot(time_list,r_list,'g',label="r(t)")
plt.plot(time_list, d_list, "k", label="d(t)")
peak_r = time_list[np.argmax(i_list)]
plt.axvline(peak_r, color='gray', linestyle='--', label="Peak Infection")
plt.ylim((0, 1)) 
plt.title("SIRD Model w/ Recursion")
plt.xlabel("Time (days)")
plt.ylabel("Fraction of Population")
plt.legend()
plt.show()

plt.figure()

# odeint
plt.plot(t, S, "r--", label="S (odeint)")
plt.plot(t, I, "b--", label="I (odeint)")
plt.plot(t, R, "g--", label="R (odeint)")
plt.plot(t, D, "k--", label="D (odeint)")  

# recurszion
plt.plot(time_list, s_list, "r", label="S (recursion)")
plt.plot(time_list, i_list, "b", label="I (recursion)")
plt.plot(time_list, r_list, "g", label="R (recursion)")
plt.plot(time_list, d_list, "k", label="D (recursion)")

plt.title("SIRD Model: odeint vs Recursion")
plt.xlabel("Time (days)")
plt.ylabel("Fraction of Population")
plt.legend()
plt.ylim(0, 1)
plt.grid(True)
plt.show()
