from mc_control import *
import matplotlib.pyplot as plt


def sarsa(iterations,lambd,Q_star):
    Q = np.zeros((2,22,11))
    Nsa = np.zeros((2,22,11))
    N0 = 100
    mse = []

    for episode in range(iterations):
        Esa = np.zeros((2, 22, 11))
        s = State()
        eps = N0 / (N0 + np.sum(Nsa[:, s.player, s.dealer], axis=0))
        a = eps_greedy_policy(Q, eps, s)
        while not s.is_terminal:
            Nsa[a,s.player,s.dealer] += 1
            Esa[a,s.player,s.dealer] += 1
            r,next_s = step(s,a)

            if next_s.is_terminal:
                delta = r - Q[a,s.player,s.dealer]
                next_a = None
            else:
                eps = N0 / (N0 + np.sum(Nsa[:, next_s.player, next_s.dealer], axis=0))
                next_a = eps_greedy_policy(Q, eps, next_s)
                delta = r + Q[next_a,next_s.player,next_s.dealer] - Q[a,s.player,s.dealer]

            alpha = 1 / Nsa[a, s.player, s.dealer]
            Q += alpha * delta * Esa
            Esa *= lambd
            s = next_s
            a = next_a

        mse.append(np.square(Q_star[:,1:,1:]-Q[:,1:,1:]).sum())

    return Q,mse


def main():
    Q_star = mc_control(100000)
    learning_curves = []
    for lambd in np.arange(0,1.01,0.1):
        print (lambd)
        _,mse = sarsa(100000, lambd, Q_star)
        learning_curves.append(mse)

    plt.plot(learning_curves[0],label='lambda=0')
    plt.plot(learning_curves[-1],label='lambda=1')
    plt.legend(loc='best')
    plt.xlabel('iteration')
    plt.ylabel('mse')
    plt.savefig('figures/sarsa_learning_curve')

    plt.figure()
    plt.plot(np.arange(0,1.01,0.1),[x[-1] for x in learning_curves],'o')
    plt.xlabel('lambda')
    plt.ylabel('mse')
    plt.savefig('figures/sarsa_lambda')








if __name__=='__main__':
    main()