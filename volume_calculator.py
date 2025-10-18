import cv2
import numpy as np
import open3d as o3d

def detect_reference_scale(img_path, known_length_cm=10):
    """
    Automatically detect a rectangular ruler / reference object in the image
    Returns pixels per cm scale
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    ref_contour = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if area > max_area and 50 < w < 1000 and 5 < h < 500:
            max_area = area
            ref_contour = cnt

    if ref_contour is None:
        return 1  # fallback scale

    x, y, w, h = cv2.boundingRect(ref_contour)
    pixel_length = max(w, h)
    scale = known_length_cm / pixel_length  # cm per pixel
    return scale

def preprocess_images_auto_scale(image_paths, known_length_cm=10):
    silhouettes = []
    scales = []

    for img_path in image_paths:
        scale = detect_reference_scale(img_path, known_length_cm)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (500, 500))
        _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        silhouettes.append(mask)
        scales.append(scale)
    return silhouettes, scales

def generate_3d_point_cloud(silhouettes, scales):
    points = []
    for i, sil in enumerate(silhouettes):
        ys, xs = np.where(sil > 0)
        zs = np.full(xs.shape, i)
        scale = scales[i]
        pts = np.stack([xs*scale, ys*scale, zs*scale], axis=1)
        points.extend(pts)
    return np.array(points)

def reconstruct_mesh(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, alpha=5.0)
    mesh.compute_vertex_normals()
    return mesh

def calculate_volume(mesh):
    try:
        return mesh.get_volume()
    except:
        hull, _ = mesh.compute_convex_hull()
        return hull.get_volume()

def estimate_volume_auto(image_paths, pattern_type, known_length_cm=10):
    silhouettes, scales = preprocess_images_auto_scale(image_paths, known_length_cm)
    points = generate_3d_point_cloud(silhouettes, scales)
    mesh = reconstruct_mesh(points)
    volume = calculate_volume(mesh)

    if pattern_type == "two":
        volume *= 1.8
    elif pattern_type == "split":
        volume *= 2.5

    return round(volume, 2), mesh

def save_reference_preview(img_path, output_path, known_length_cm=10):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    ref_contour = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if area > max_area and 50 < w < 1000 and 5 < h < 500:
            max_area = area
            ref_contour = cnt

    preview = img.copy()
    if ref_contour is not None:
        x, y, w, h = cv2.boundingRect(ref_contour)
        cv2.rectangle(preview, (x,y), (x+w, y+h), (0,255,0), 3)

    cv2.imwrite(output_path, preview)
    return output_path
