import math

# Параметры 
t0 = 0.0
t_end = 0.1
h = 0.01
y0 = 0.5

# y' = f(t,y)
def f(t, y):
    return math.cos(t) - y

# y(t) = (sin t + cos t)/2
def y_exact(t):
    return 0.5 * (math.sin(t) + math.cos(t))

# RK4 po usloviyu
def rk4_step(t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2.0, y + h*k1/2.0)
    k3 = f(t + h/2.0, y + h*k2/2.0)
    k4 = f(t + h, y + h*k3)
    y_next = y + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    t_next = t + h
    return t_next, y_next

# сетка численного решения
n_steps = int(round((t_end - t0) / h))
ts = [t0]
ys = [y0]

t = t0
y = y0
for _ in range(n_steps):
    t, y = rk4_step(t, y, h)
    ts.append(t)
    ys.append(y)

# Вывод таблицы
header = "Решение y' = cos(t) - y, интервал [{:.2f}; {:.2f}], h = {:.2f}, метод = RK4, y(0) = {}".format(t0, t_end, h, y0)
print(header)
print("-" * len(header))
print("{:>2} {:>6} {:>14} {:>14} ".format("i", "t", "y_numeric", "y_exact"))

max_err = 0.0
for i, (t, y_num) in enumerate(zip(ts, ys)):
    y_ex = y_exact(t)
    err = abs(y_num - y_ex)
    if err > max_err:
        max_err = err
    print("{:2d} {:6.2f} {:14.9f} {:14.9f}".format(i, t, y_num, y_ex ))

