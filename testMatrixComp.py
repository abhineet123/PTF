import numpy as np
import time
import sys
from scipy.linalg import expm
from Misc import printMatrixToFile, printVectorFile, printScalarToFile



def writeVectorToFile(filename, vec):
    fid = open(filename, 'r')
    vec_size = vec.size
    for i in xrange(vec_size):
        fid.write('{:16.12f}\n'.format(vec[i]))
    fid.close()


if __name__ == '__main__':
    mat_sizes = range(5, 205, 5)
    mat_count = len(mat_sizes)

    inv_times = []
    transpose_times = []
    prod_times = []
    det_times = []
    lslu_times = []
    solve_times = []
    exp_times = []

    for i in xrange(mat_count):
        mat_size = mat_sizes[i]
        fname1 = 'matrices/mat{:d}_1.txt'.format(mat_size)
        fname2 = 'matrices/mat{:d}_2.txt'.format(mat_size)
        # mat1 = np.fromfile(fname1, dtype=np.float64, sep='\t').reshape((mat_size, mat_size))
        # mat2 = np.fromfile(fname2, dtype=np.float64, sep='\t').reshape((mat_size, mat_size))

        mat1 = np.loadtxt(fname1, dtype=np.float64, delimiter='\t')
        mat2 = np.loadtxt(fname2, dtype=np.float64, delimiter='\t')
        vec1 = mat2[:, 0]

        # print 'mat_size: ', mat_size
        # print 'mat1.shape:', mat1.shape
        # print 'mat2.shape:', mat2.shape

        fname = 'matrices/mat{:d}_result_np.txt'.format(mat_size)

        start_time = time.clock()
        mat1_inv = np.linalg.inv(mat1)
        mat2_inv = np.linalg.inv(mat2)
        end_time = time.clock()
        inv_time = end_time - start_time
        inv_times.append(inv_time)
        printMatrixToFile(mat1_inv, 'mat1_inv', fname, '{:16.12f}', 'w')
        printMatrixToFile(mat2_inv, 'mat2_inv', fname, '{:16.12f}', 'a')
        # np.savetxt('matrices/mat{:d}_1_inv_np.txt'.format(mat_size), mat1_inv, delimiter='\t', fmt='%16.12f')
        # np.savetxt('matrices/mat{:d}_2_inv_np.txt'.format(mat_size), mat2_inv, delimiter='\t', fmt='%16.12f')

        start_time = time.clock()
        mat1_t = np.transpose(mat1)
        mat2_t = np.transpose(mat2)
        end_time = time.clock()
        transpose_time = end_time - start_time
        transpose_times.append(transpose_time)
        printMatrixToFile(mat1_t, 'mat1_transpose', fname, '{:16.12f}', 'a')
        printMatrixToFile(mat2_t, 'mat2_transpose', fname, '{:16.12f}', 'a')
        # np.savetxt('matrices/mat{:d}_1_transpose_np.txt'.format(mat_size), mat1_t, delimiter='\t', fmt='%16.12f')
        # np.savetxt('matrices/mat{:d}_2_transpose_np.txt'.format(mat_size), mat2_t, delimiter='\t', fmt='%16.12f')

        start_time = time.clock()
        mat12 = np.dot(mat1, mat2)
        end_time = time.clock()
        prod_time = end_time - start_time
        prod_times.append(prod_time)
        printMatrixToFile(mat12, 'mat_prod', fname, '{:16.12f}', 'a')
        # np.savetxt('matrices/mat{:d}_prod_np.txt'.format(mat_size), mat12, delimiter='\t', fmt='%16.12f')

        start_time = time.clock()
        mat1_det = np.linalg.det(mat1)
        mat2_det = np.linalg.det(mat2)
        end_time = time.clock()
        det_time = end_time - start_time
        det_times.append(det_time)
        # printMatrixToFile(mat1_det, 'mat1_det', fname, '{:16.12f}', 'a')
        # printMatrixToFile(mat2_det, 'mat2_det', fname, '{:16.12f}', 'a')

        start_time = time.clock()
        vec2, residuals, rank, s = np.linalg.lstsq(mat1, vec1)
        end_time = time.clock()
        lslu_time = end_time - start_time
        lslu_times.append(lslu_time)
        printVectorFile(vec2, 'mat_lslu', fname, '{:16.12f}', 'a', '\n')
        # np.savetxt('matrices/mat{:d}_lslu_np.txt'.format(mat_size), vec2, delimiter='\t', fmt='%16.12f')

        start_time = time.clock()
        vec3 = np.linalg.solve(mat1, vec1)
        end_time = time.clock()
        solve_time = end_time - start_time
        solve_times.append(solve_time)
        printVectorFile(vec3, 'mat_solve', fname, '{:16.12f}', 'a', '\n')
        # np.savetxt('matrices/mat{:d}_solve_np.txt'.format(mat_size), vec3, delimiter='\t', fmt='%16.12f')

        start_time = time.clock()
        mat1_exp = expm(mat1)
        mat2_exp = expm(mat2)
        end_time = time.clock()
        exp_time = end_time - start_time
        exp_times.append(exp_time)
        printMatrixToFile(mat1_exp, 'mat1_exp', fname, '{:16.12f}', 'a')
        printMatrixToFile(mat2_exp, 'mat2_exp', fname, '{:16.12f}', 'a')

        sys.stdout.write('mat_size: {:4d}\t'.format(mat_size))
        sys.stdout.write('inv_time: {:15.12f}\t'.format(inv_time))
        sys.stdout.write('transpose_time: {:15.12f}\t'.format(transpose_time))
        sys.stdout.write('prod_time: {:15.12f}\t'.format(prod_time))
        sys.stdout.write('det_time: {:15.12f}\t'.format(det_time))
        sys.stdout.write('lslu_time: {:15.12f}\t'.format(lslu_time))
        sys.stdout.write('solve_time: {:15.12f}\t'.format(solve_time))
        sys.stdout.write('exp_time: {:15.12f}\t'.format(exp_time))
        sys.stdout.write('\n')

        # print 'mat_size: {:4d}\t inv_time: {:15.12f}\t transpose_time: {:15.12f}\t prod_time: {:15.12f}\t det_time: {:15.12f}\t lslu_time={:15.12f}\t solve_times={:15.12f}'.format(
        # mat_size, inv_time, transpose_time, prod_time, det_time, lslu_time, solve_time)

    times = [inv_times, transpose_times, prod_times, det_times, lslu_times, solve_times, exp_times]
    times_arr = np.array(times, dtype=np.float64).transpose()
    # print 'times_arr.shape:', times_arr.shape
    np.savetxt('matrices/times_np.txt', times_arr, fmt='%16.12f', delimiter='\t')
    # times_arr.tofile('matrices/times_np.txt', sep='\t', format='%12.8f')



