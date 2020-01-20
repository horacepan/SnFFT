import pdb
import argparse
import numpy as np

from s8cube import S8CubeEnv
from memory import ReplayMemory

def compute_grad(y, X, w):
    '''
    Compute grad of J(w) = ||y - Xw||_2^2:
    \grad_w(J) = y^Ty - 2y^T (Xw) + (w^T X^ X w)
               = 0 - 2y^TX + 2X^TXw
    y: numpy vector of dim n x 1
    X: numpy matrix of dim n x d
    w: numpy vector of dim d x 1
    Returns numpy vector of dim (shape w)
    d x n x n x 1 + d x n x n x d x d x 1 => d x 1
    '''
    #print(f'y shape: {y.shape}')
    #print(f'X shape: {X.shape}')
    #print(f'w shape: {w.shape}')
    return (-2 * X.T.dot(y) + 2 * X.T.dot(X).dot(w))
    #p1 = (-2 * X.T.dot(y))
    #p2 = 2 * X.T.dot(X).dot(w)
    #print(f'p1 shape: {p1.shape}')
    #print(f'p2 shape: {p2.shape}')
    #return p1 + p2

# first can we learn it given the true distances?
# then can we learn it under RL framework?
def main(args):
    env = S8CubeEnv(nvecs=args.nvecs)
    memory = ReplayMemory(args.capacity, args.nvecs)
    theta = np.random.random((env.nvecs, 1)) # no bias?

    losses = []
    print('Starting')
    for e in range(args.epochs):
        state = env.reset()
        for i in range(args.maxep_len):
            state_idx = env.idx_state
            action = env.get_opt_action(env.tup_state, theta) 
            new_state, rew, done, _ = env.step(action)
            memory.push(state_idx,
                        action,
                        env.idx_state,
                        rew,
                        done)
            #memory.push(state, action, next_state, reward, done)
            # would rather push the index which can be converted to anything we want

        if e % args.update_int == 0:
            print('Updating')
            batch = memory.sample(args.batch_size)
            # TODO: compute expected
            expected = env.get_true_dists(batch.next_state)
            next_vecs = env.get_vecs(batch.next_state.reshape(-1))
            grad_theta = compute_grad(expected, next_vecs, theta)

            # loss is sum of sq errors
            delta = expected - next_vecs.dot(theta)
            loss = np.square(delta).sum()
            theta += (args.lr * grad_theta)
            losses.append(loss)

        # log stuff
        if e % args.log_int == 0:
            print('Epoch {:4d} | Mean loss: {:.2f} | Last {} loss: {:.2f}'.format(
                e, np.mean(losses), args.log_int, np.mean(losses[-args.log_int:])
            ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # env params
    parser.add_argument('--nvecs', type=int, default=320)

    # replay memory params
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--capacity', type=int, default=5000)

    # optimization params
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--eps_min', type=int, default=0.05)
    parser.add_argument('--update_int', type=int, default=50)
    parser.add_argument('--log_int', type=int, default=50)
    parser.add_argument('--maxep_len', type=int, default=20)
    parser.add_argument('--discount', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
