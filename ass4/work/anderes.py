def LeastSquaresGN(p_anchor, p_start, r, max_iter, tol):
    current_p = p_start
    for i in range(max_iter):
        jacobi = create_jacobi_matrix(current_p, p_anchor)
        prev_p = current_p
        current_p = np.subtract(current_p, np.dot(np.dot(np.linalg.inv(np.dot(jacobi.T, jacobi)), jacobi.T), np.subtract(r, calculate_anchor_distances(current_p[0], p_anchor))))
        change = np.subtract(current_p, prev_p)[0]
        exit_condition = math.sqrt(math.pow(change[0], 2) + math.pow(change[1], 2))
        if (exit_condition < tol):
            break;
    return current_p;

def create_jacobi_matrix(p_start, p_anchor):
    jacobi = np.zeros(((len(p_anchor), 2)))
    p_start_x = p_start[0][0]
    p_start_y = p_start[0][1]

    for i in range(len(p_anchor)):
        jacobi[i][0] = (p_anchor[i][0] - p_start_x) / calculate_euclidean_distance(p_start[0], p_anchor[i])
        jacobi[i][1] = (p_anchor[i][1] - p_start_y) / calculate_euclidean_distance(p_start[0], p_anchor[i])

    return jacobi