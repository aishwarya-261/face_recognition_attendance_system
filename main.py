import tkinter as tk
from tkinter import messagebox
import attendance_pipeline

def clear_entry(entry):
    entry.delete(0, 'end')

def take_images_gui():
    enrollment_id = txt_enroll.get()
    name = txt_name.get()
    
    if enrollment_id == "" or name == "":
        messagebox.showerror("Error", "Please enter both Enrollment ID and Name")
        return
        
    res = attendance_pipeline.capture_images(enrollment_id, name)
    if "Success" in res:
         messagebox.showinfo("Success", res)
    else:
         messagebox.showerror("Error", res)
         
def train_images_gui():
    res = attendance_pipeline.train_images()
    if "Successfully" in res:
         messagebox.showinfo("Success", res)
    else:
         messagebox.showerror("Error", res)

def automatic_attendance_gui():
    res = attendance_pipeline.automatic_attendance()
    if "saved" in res:
         messagebox.showinfo("Success", res)
    else:
         messagebox.showwarning("Notice", res)

def on_enter(e):
    e.widget['background'] = '#3b82f6'

def on_leave(e):
    e.widget['background'] = '#2563eb'

window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("850x550")
window.configure(bg="#0f172a")

# Title Frame
title_frame = tk.Frame(window, bg="#1e293b", height=90)
title_frame.pack(fill=tk.X)
title_label = tk.Label(title_frame, text="Face Recognition Attendance System", fg="#f8fafc", bg="#1e293b", font=('Helvetica', 24, 'bold'))
title_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Main Content Frame
content_frame = tk.Frame(window, bg="#0f172a")
content_frame.pack(fill=tk.BOTH, expand=True, pady=30)

# Form Section
form_frame = tk.Frame(content_frame, bg="#1e293b", padx=40, pady=40, bd=0)
form_frame.place(relx=0.5, y=100, anchor=tk.CENTER)

# Enrollment Input
lbl_enroll = tk.Label(form_frame, text="Enrollment ID:", fg="#94a3b8", bg="#1e293b", font=('Helvetica', 12, 'bold'))
lbl_enroll.grid(row=0, column=0, pady=15, sticky='w')

txt_enroll = tk.Entry(form_frame, width=25, bg="#334155", fg="white", font=('Helvetica', 14), insertbackground='white', bd=0, highlightthickness=1, highlightcolor="#3b82f6", highlightbackground="#475569")
txt_enroll.grid(row=0, column=1, padx=20, pady=15, ipady=6)

btn_clear_enroll = tk.Button(form_frame, text="Clear", command=lambda: clear_entry(txt_enroll), fg="white", bg="#ef4444", bd=0, activebackground="#dc2626", font=('Helvetica', 10, 'bold'), cursor="hand2")
btn_clear_enroll.grid(row=0, column=2, padx=5, ipady=4, ipadx=10)

# Name Input
lbl_name = tk.Label(form_frame, text="Student Name:", fg="#94a3b8", bg="#1e293b", font=('Helvetica', 12, 'bold'))
lbl_name.grid(row=1, column=0, pady=15, sticky='w')

txt_name = tk.Entry(form_frame, width=25, bg="#334155", fg="white", font=('Helvetica', 14), insertbackground='white', bd=0, highlightthickness=1, highlightcolor="#3b82f6", highlightbackground="#475569")
txt_name.grid(row=1, column=1, padx=20, pady=15, ipady=6)

btn_clear_name = tk.Button(form_frame, text="Clear", command=lambda: clear_entry(txt_name), fg="white", bg="#ef4444", bd=0, activebackground="#dc2626", font=('Helvetica', 10, 'bold'), cursor="hand2")
btn_clear_name.grid(row=1, column=2, padx=5, ipady=4, ipadx=10)

# Hover effects for clear buttons
def bind_danger_hover(btn):
    btn.bind("<Enter>", lambda e: e.widget.config(bg="#f87171"))
    btn.bind("<Leave>", lambda e: e.widget.config(bg="#ef4444"))

bind_danger_hover(btn_clear_enroll)
bind_danger_hover(btn_clear_name)

# Action Buttons Frame
action_frame = tk.Frame(content_frame, bg="#0f172a")
action_frame.place(relx=0.5, y=300, anchor=tk.CENTER)

def create_primary_btn(parent, text, command):
    btn = tk.Button(parent, text=text, command=command, fg="white", bg="#2563eb", bd=0, activebackground="#1d4ed8", font=('Helvetica', 13, 'bold'), cursor="hand2")
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn

btn_take = create_primary_btn(action_frame, "1. Take Images", take_images_gui)
btn_take.grid(row=0, column=0, padx=15, ipady=12, ipadx=20)

btn_train = create_primary_btn(action_frame, "2. Train Model", train_images_gui)
btn_train.grid(row=0, column=1, padx=15, ipady=12, ipadx=20)

btn_attend = create_primary_btn(action_frame, "3. Automatic Attendance", automatic_attendance_gui)
btn_attend.grid(row=0, column=2, padx=15, ipady=12, ipadx=20)
btn_attend.config(bg="#10b981", activebackground="#059669")
btn_attend.bind("<Enter>", lambda e: e.widget.config(bg="#34d399"))
btn_attend.bind("<Leave>", lambda e: e.widget.config(bg="#10b981"))

if __name__ == "__main__":
    attendance_pipeline.ensure_folders()
    window.mainloop()
