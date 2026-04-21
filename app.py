import gradio as gr
import attendance_pipeline
import os
from PIL import Image

def enroll_student(enrollment_id, name, image):
    if image is None:
        return "Please upload or take a photo."
    # image is a PIL image from Gradio
    res = attendance_pipeline.enroll_from_image(enrollment_id, name, image, 1)
    return res

def train_model():
    res = attendance_pipeline.train_images()
    return res

def mark_attendance(image):
    if image is None:
        return "Please upload or take a photo."
    status, result = attendance_pipeline.recognize_face(image)
    return f"Status: {status}\nResult: {result}"

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 Face Recognition Attendance System")
    gr.Markdown("Register students, train the AI model, and mark attendance using face recognition.")
    
    with gr.Tab("1. Enroll Student"):
        with gr.Row():
            with gr.Column():
                enroll_id = gr.Textbox(label="Enrollment ID", placeholder="e.g. 101")
                student_name = gr.Textbox(label="Student Name", placeholder="e.g. John Doe")
                input_img = gr.Image(type="pil", label="Capture/Upload Face")
                enroll_btn = gr.Button("Register Student", variant="primary")
            with gr.Column():
                enroll_output = gr.Textbox(label="Output")
        
        enroll_btn.click(enroll_student, inputs=[enroll_id, student_name, input_img], outputs=enroll_output)

    with gr.Tab("2. Train Model"):
        gr.Markdown("Click the button below to train the model on all enrolled students.")
        train_btn = gr.Button("🚀 Start Training", variant="primary")
        train_output = gr.Textbox(label="Training Status")
        train_btn.click(train_model, outputs=train_output)

    with gr.Tab("3. Mark Attendance"):
        with gr.Row():
            with gr.Column():
                attendance_img = gr.Image(type="pil", label="Capture/Upload Face for Attendance")
                attendance_btn = gr.Button("✅ Mark Attendance", variant="primary")
            with gr.Column():
                attendance_output = gr.Textbox(label="Attendance Result")
        
        attendance_btn.click(mark_attendance, inputs=attendance_img, outputs=attendance_output)

    gr.Markdown("---")
    gr.Markdown("Built for automated attendance using FaceNet and MTCNN.")

if __name__ == "__main__":
    attendance_pipeline.ensure_folders()
    demo.launch()
