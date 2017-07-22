from environment import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def eps_greedy_policy(q,eps,s):
    if random.random()<eps:
        return random.choice([HIT,STICK])

    return np.argmax(q[:,s.player,s.dealer])


def mc_control(iterations):
    Q = np.zeros((2,22,11))
    Nsa = np.zeros((2,22,11))
    N0 = 100

    for i in range(iterations):
        if i%100000==0:
            print(i)

        visits = []
        s = State()
        total_r = 0
        while not s.is_terminal:
            eps = N0 / (N0 + np.sum(Nsa[:,s.player,s.dealer],axis=0))
            a = eps_greedy_policy(Q,eps,s)
            Nsa[a,s.player,s.dealer] += 1
            r,next_s = step(s,a)
            total_r += r
            alpha = 1 / Nsa[a, s.player, s.dealer]
            visits.append((s,a,alpha))
            s = next_s

        for s_t,a_t,alpha_t in visits:
            Q[a_t,s_t.player,s_t.dealer] += alpha_t * (total_r - Q[a_t,s_t.player,s_t.dealer])

    return Q


def plot(V,file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = range(10)
    y = range(21)
    X, Y = np.meshgrid(x, y)
    ax.plot_wireframe(X + 1, Y + 1, V[1:, 1:])
    ax.set_xlabel("dealer starting card")
    ax.set_ylabel("player current sum")
    ax.set_zlabel("value of state")
    plt.savefig('figures/'+file_name)



if __name__ == "__main__":
    Q = mc_control(1000000)
    V = np.max(Q,axis=0)
    plot(V,'mc.png')