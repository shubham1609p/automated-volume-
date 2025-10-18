from flask import Flask, render_template, request, send_file, jsonify, url_for
import os
from fpdf import FPDF
from volume_calculator import estimate_volume_auto
import open3d as o3d

app = Flask(__name__, static_url_path='/static', static_folder='uploads')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    pattern_type = request.form['patternType']

    # Save uploaded images
    images = request.files.getlist('images')
    saved_files = []
    for img in images:
        path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(path)
        saved_files.append(path)

    # Estimate volume automatically using detected reference object
    estimated_volume, mesh = estimate_volume_auto(saved_files, pattern_type, known_length_cm=10)

    # Save mesh as GLB for Three.js viewer
    mesh_path = os.path.join(UPLOAD_FOLDER, "mesh.glb")
    o3d.io.write_triangle_mesh(mesh_path, mesh, write_triangle_uvs=True, write_vertex_normals=True, write_vertex_colors=True)

    # Generate PDF report
    pdf_path = generate_pdf(saved_files, estimated_volume, pattern_type, mesh_path)

    return jsonify({
        "volume": estimated_volume,
        "mesh_url": url_for('static', filename='mesh.glb'),
        "pdf_url": url_for('static', filename='volume_report.pdf')
    })

def generate_pdf(images, volume, pattern_type, mesh_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Casting Pattern Volume Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Pattern Type: {pattern_type}", ln=True)
    pdf.cell(0, 10, f"Estimated Volume: {volume} cm³", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Uploaded Images:", ln=True)
    pdf.ln(5)
    for img_path in images:
        pdf.image(img_path, w=60)
        pdf.ln(5)
    pdf.ln(5)
    pdf.cell(0, 10, f"3D Mesh saved as: {mesh_path}", ln=True)
    pdf_path = os.path.join(UPLOAD_FOLDER, "volume_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

if __name__ == "__main__":
    app.run(debug=True)
