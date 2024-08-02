#-*-coding:utf-8-*-
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pyclipper
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def k_means(data, k, max_iter=100):
    centers = {}
    n_data = data.shape[0]
    for idx, i in enumerate(random.sample(range(n_data), k)):
        centers[idx] = data[i]

    for i in range(max_iter):
        clusters = {}
        for j in range(k):
            clusters[j] = []

        for sample in data:
            distances = []
            for c in centers:
                distances.append(distance(sample, centers[c]))
            idx = np.argmin(distances)
            clusters[idx].append(sample)

        pre_centers = centers.copy()

        for c in clusters.keys():
            centers[c] = np.mean(clusters[c], axis=0)

        is_convergent = True
        for c in centers:
            if distance(pre_centers[c], centers[c]) > 1e-8:  # 中心点是否变化
                is_convergent = False
                break
        if is_convergent == True:

            break
    return centers, clusters


def predict(p_data, centers):
    distances = [distance(p_data, centers[c]) for c in centers]
    return np.argmin(distances)

def dis(waypoint, clusters,k):
    mindis =[]
    for i in range(k):
        cluster = np.array(clusters[i])
        sortd =[]
        for j in range(cluster.shape[0]):
            d= np.sum((waypoint - cluster[j,:]) ** 2)
            sortd.append(d)
        mindis.append(min(sortd))
    closests = np.argmin(np.array(mindis), axis=0)

    return np.array(clusters[closests])

def equidistant_zoom_contour(contour, margin):
    """
    等距离缩放多边形轮廓点
    :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
    :param margin: 轮廓外扩的像素距离，margin正数是外扩，负数是缩小
    :return: 外扩后的轮廓点
    """
    pco = pyclipper.PyclipperOffset()
    pco.MiterLimit = 10
    pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    solution = np.array(solution)
    solution = solution.squeeze(0)

    return solution

def is_in_line(point, o, d):

    if o[1] > point[1] and d[1] > point[1]:
        return False
    if o[1] < point[1] and d[1] < point[1]:
        return False
    if o[0] > point[0] and d[0] > point[0]:
        return False
    if o[0] < point[0] and d[0] < point[0]:
        return False

    if o[0] == d[0]:
        return True if point[0] == o[0] else False

    a = (d[1] - o[1]) / (d[0] - o[0])
    b = (d[0] * o[1] - o[0] * d[1]) / (d[0] - o[0])
    y = a * point[0] + b
    return True if y == point[1] else False

def delete_points(X, Y, E_points):
    x = X.reshape(X.shape[0]* X.shape[1],-1)
    y = Y.reshape(Y.shape[0] * Y.shape[1],-1)
    points = np.concatenate((x,y),axis=1)
    # data = points

    area = E_points
    polygon = Polygon(area)
    i = 0
    index = []

    for point in points:
        point1 = Point(point)
        in_contain = polygon.contains(point1)

        in_line = False
        for j in range(len(area) - 1):
            if in_line:
                break
            origin_coord = area[j, :]
            dest_coord = area[j + 1, :]
            in_line = is_in_line(point, origin_coord, dest_coord)


        if in_contain == False & in_line == False:
            index.append(i)
        i = i + 1
    data = np.delete(points, index, 0)

    return data

def reGen_points(pregendata, waypoint, r, interval, k):

    #step1: cluster
    centers, clusters = k_means(pregendata, k)

    #step2: waypoint in the cluster
    cluster = dis(waypoint, clusters,k)

    #step3: initial region in the cluster
    hull = ConvexHull(cluster)

    hull1 = hull.vertices.tolist()
    hull1.append(hull1[0])
    hull1_points = pregendata[hull1, :]


  #step4: greedy region
    expand_points = equidistant_zoom_contour(hull1_points, r)
    E_points = np.zeros((expand_points.shape[0] + 1, 2))
    E_points[:-1, :] = expand_points
    E_points[-1, :] = expand_points[0, :]


    #step5: points regeneration in greedy point
    max_x = np.max(E_points, axis=0)[0]
    min_x = np.min(E_points, axis=0)[0]
    max_y = np.max(E_points, axis=0)[1]
    min_y = np.min(E_points, axis=0)[1]

    numx =int((max_x-min_x)/interval)
    numy =int((max_y-min_y)/interval)
    x = np.linspace(min_x, max_x, numx)
    y = np.linspace(min_y, max_y, numy)
    X, Y = np.meshgrid(x, y)


    #step6: delete points
    data = delete_points(X, Y, E_points)

    data = np.concatenate((data,E_points), 0)
    data = np.unique(data,axis=0)

    return data


def Regenerate_points(pregendata, waypoint, r, interval,k,magnitude):
    re_points =[]
    for i in range(pregendata.shape[1]):
        preway =pregendata[:,i,:]
        gtway= waypoint[i,:]
        regenerated_points = reGen_points(preway, gtway, r, interval,k)

        All_repoints=np.concatenate((regenerated_points, preway),axis=0)
        l2error_sample = np.linalg.norm(All_repoints - gtway, axis=1)
        indices = np.argmin(l2error_sample, axis=0)
        regenerated_points1 =All_repoints[indices,:]

        re_points.append(regenerated_points1/magnitude)

    return re_points

