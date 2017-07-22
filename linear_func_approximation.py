from td_learning import *


def compute_features(player,dealer,a):
    dealer_intervals = [[1, 4], [4, 7], [7, 10]]
    player_intervals = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
    actions = [HIT,STICK]

    features = np.zeros(len(dealer_intervals)*len(player_intervals)*len(actions))
    i = 0
    for dealer_int in dealer_intervals:
        for player_int in player_intervals:
            for action in actions:
                if dealer_int[0]<= dealer <= dealer_int[1] \
                        and player_int[0] <= player <= player_int[1] \
                        and action == a:
                    features[i] = 1

                i += 1

    return features

def compute_phi():
    phi = {}
    for p in range(1,22):
        for d in range(1,11):
            for a in [HIT,STICK]:
                phi[(p,d,a)] = compute_features(p,d,a)
    return phi


def q_hat(s, a, w,phi):
    f = phi[(s.player,s.dealer,a)]
    return np.sum(f*w)


def eps_greedy_policy(eps,s,w,phi):
    if random.random()<eps:
        return random.choice([HIT,STICK])

    hit_q = q_hat(s,HIT,w,phi)
    stick_q = q_hat(s,STICK,w,phi)

    if hit_q>stick_q:
        return HIT
    return STICK


def sarsa_linear_approx(iterations,lambd,Q_star):
    w = np.random.normal(0,0.01,36)
    mse = []
    eps = 0.05
    alpha = 0.01
    phi = compute_phi()

    for episode in range(iterations):
        Esa = np.zeros(36)
        s = State()
        a = eps_greedy_policy(eps, s, w,phi)
        while not s.is_terminal:
            r,next_s = step(s,a)

            if next_s.is_terminal:
                delta = r - q_hat(s,a,w,phi)
                next_a = None
            else:
                next_a = eps_greedy_policy(eps, next_s, w,phi)
                delta = r + q_hat(next_s,next_a,w,phi) - q_hat(s,a,w,phi)

            Esa = lambd*Esa + phi[(s.player,s.dealer,a)]
            w_delta = alpha * delta * Esa
            w += w_delta

            s = next_s
            a = next_a

        m = 0
        for d in range(1,11):
            for p in range(1,22):
                for a in [HIT,STICK]:
                    s.dealer = d
                    s.player = p
                    m += (Q_star[a,p,d] - q_hat(s,a,w,phi))**2
        mse.append(m)

    return mse


def main():
    Q_star = mc_control(100000)
    learning_curves = []
    for lambd in np.arange(0,1.01,0.1):
        print (lambd)
        mse = sarsa_linear_approx(10000, lambd, Q_star)
        learning_curves.append(mse)

    plt.plot(learning_curves[0],label='lambda=0')
    plt.plot(learning_curves[-1],label='lambda=1')
    plt.legend(loc='best')
    plt.xlabel('iteration')
    plt.ylabel('mse')
    plt.savefig('figures/sarsa_linear_learning_curve')

    plt.figure()
    plt.plot(np.arange(0,1.01,0.1),[x[-1] for x in learning_curves],'o')
    plt.xlabel('lambda')
    plt.ylabel('mse')
    plt.savefig('figures/sarsa_linear_lambda')


if __name__=='__main__':
    main()
