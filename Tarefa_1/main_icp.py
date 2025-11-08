import cv2
import numpy as np
import open3d as o3d
from copy import deepcopy

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 1.622727601414635, 0.92732282434191027, 2.9839999675750732 ],
			"boundingbox_min" : [ -1.7043713967005412, -1.1394533466157459, -0.029999999999999999 ],
			"field_of_view" : 60.0,
			"front" : [ 0.54877655571863571, 0.040714697251322977, -0.83497700885792325 ],
			"lookat" : [ 0.39833293954817622, -0.11328460994204304, 1.9350827314289289 ],
			"up" : [ -0.2163456823136507, -0.95786868355247656, -0.18889714347677763 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


def main():

    # =========================================================
    # 1) CARREGAMENTO (OpenCV) + FILTRAGEM DE PROFUNDIDADE
    # =========================================================
    # Imagem 1 (RGB, Depth)
    rgb1_cv  = cv2.cvtColor(cv2.imread("tum_dataset/rgb/1.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth1   = cv2.imread("tum_dataset/depth/1.png", cv2.IMREAD_UNCHANGED)  # Não alterar o número de bits (PNG de 16 bits)
    depth1   = cv2.medianBlur(depth1, 5)                                    # Filtro de mediana para reduzir ruído, compara cada píxel com a vizinhança numa matriz 5x5
    depth1_m = depth1.astype(np.float32) / 5000.0                           # uint16 para float32 e de milímetros para metros (Fator de escala assumido:5000)
    depth1_m[(depth1_m < 0.1) | (depth1_m > 3.0)] = 0.0                     # Remove valores de profundidade inválidos (< 0.1m ou > 3.0m), '|' é o operador 'ou' em numpy

    # Imagem 2 (RGB, Depth)
    rgb2_cv  = cv2.cvtColor(cv2.imread("tum_dataset/rgb/2.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth2   = cv2.imread("tum_dataset/depth/2.png", cv2.IMREAD_UNCHANGED)
    depth2   = cv2.medianBlur(depth2, 5)
    depth2_m = depth2.astype(np.float32) / 5000.0
    depth2_m[(depth2_m < 0.1) | (depth2_m > 3.0)] = 0.0


    # =========================================================
    # 2) CRIAÇÃO DE NUVENS (Open3D a partir de dados OpenCV)
    # =========================================================
    color1_o3d = o3d.geometry.Image(rgb1_cv)
    depth1_o3d = o3d.geometry.Image(depth1_m)
    rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color1_o3d, depth1_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
    )

    color2_o3d = o3d.geometry.Image(rgb2_cv)
    depth2_o3d = o3d.geometry.Image(depth2_m)
    rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color2_o3d, depth2_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
    )

    intr = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, intr)
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, intr)


    # =========================================================
    # 3) PRÉ-PROCESSAMENTO
    # =========================================================
    pcd1_ds = pcd1.voxel_down_sample(0.01)  # 1 cm
    pcd1_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))  # Superfície local de 30 pontos máx num raio de 5 cm

    pcd2_ds = pcd2.voxel_down_sample(0.005)  # 0.5 cm
    pcd2_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    pcd1_ds, _ = pcd1_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd2_ds, _ = pcd2_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

    # Visualização ANTES (fonte vermelha, alvo azul)
    axes = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)
    pcd1_ds.paint_uniform_color([1, 0, 0])  # fonte: vermelho
    pcd2_ds.paint_uniform_color([0, 0, 1])  # alvo: azul
    o3d.visualization.draw_geometries(
        [pcd1_ds, pcd2_ds, axes],
        front=view['trajectory'][0]['front'],
        lookat=view['trajectory'][0]['lookat'],
        up=view['trajectory'][0]['up'],
        zoom=view['trajectory'][0]['zoom'],
    )


    # =========================================================
    # 4) ICP (Open3D) — Point-to-Plane, T_init = Semelhante à Identidade
    # =========================================================
    threshold = 0.05  # em metros
    # Transformação rígida sem rotação, com ligeira translação (x=-0.20m, y=0.05m, z=0.05m)
    T_init = np.asarray([
        [ 0.9848,  0.0872,  -0.1736,   0.80],
        [ 0.0,     1.0,      0.0,     -0.05],
        [ 0.1736,  0.0,      0.9848,  -0.05],
        [ 0.0,     0.0,      0.0,      1.00]
    ], dtype=float)

    best = None  # (rmse, fitness, threshold, result) -- rmse é o erro quadrático médio dos inliers, fitness é a razão de inliers sobre pontos da fonte

    for i in range(20):
        
        res = o3d.pipelines.registration.registration_icp(
            pcd1_ds, pcd2_ds, threshold, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
        )
        print(f"[ICP p2plane][iter {i+1:02d}] th={threshold:.3f}  fitness={res.fitness:.4f}  rmse={res.inlier_rmse:.6f}")
        # guardar melhor até agora
        if (best is None) or (res.inlier_rmse < best[0]):
            best = (res.inlier_rmse, res.fitness, threshold, res)
             # warm start para a próxima iteração
        T_init = res.transformation

    # melhor resultado
    rmse_best, fit_best, th_best, result = best
    print("\n>>> MELHOR (p2plane): th =", th_best, " fitness =", fit_best, " rmse =", rmse_best)
    print("transformation:\n", result.transformation)

    # Visualização após alinhar com o melhor resultado
    pcd1_aligned = deepcopy(pcd1_ds)  # Criar uma cópia da fonte (pcd1) para transformar
    pcd1_aligned.transform(result.transformation)
    # pcd1_aligned.transform(T_init)  # Usar T_init como resultado do ICP para esta versão

    axes = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)
    pcd1_aligned.paint_uniform_color([1,0,0])  # fonte alinhada (vermelho)
    pcd2_ds.paint_uniform_color([0,0,1])       # alvo original (azul)
    
    # Visualizar a fonte alinhada (vermelha) sobreposta ao alvo original (azul)
    o3d.visualization.draw_geometries(
        [pcd1_aligned, pcd2_ds, axes],
        front=view['trajectory'][0]['front'],
        lookat=view['trajectory'][0]['lookat'],
        up=view['trajectory'][0]['up'],
        zoom=view['trajectory'][0]['zoom'],
    )


if __name__ == "__main__":
    main()