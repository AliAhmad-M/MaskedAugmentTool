import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import os
import shutil
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
import imgaug.augmenters as iaa
import cv2
import random
import copy

class YOLOMaskAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Mask Annotator")
        self.root.geometry("1200x675")
        
        # Data storage
        self.images = []
        self.labels = []
        self.image_paths = []
        self.label_paths = []
        self.current_image_idx = 0
        
        self.mask_images = []
        self.mask_paths = []
        self.augmented_masks = []  # Store augmented versions
        
        # Class management for masks
        self.mask_classes = {}  # {class_id: {'name': str, 'masks': [images], 'augmented': [images]}}
        self.current_class = 0
        
        # Drawing state
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.selected_mask_idx = None
        
        # Editing state
        self.editing_mode = False
        self.selected_bbox_idx = None
        self.bbox_handles = []
        self.dragging_bbox = False
        self.dragging_resize = False
        self.resize_corner = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Undo stack
        self.undo_stack = []
        self.max_undo = 50
        
        # Display state
        self.display_image = None
        self.photo_image = None
        self.canvas_scale = 1.0
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.original_img_width = 0
        self.original_img_height = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Store the last canvas offsets for consistent positioning
        self.last_canvas_offset_x = 0
        self.last_canvas_offset_y = 0
        self.last_canvas_width = 0
        self.last_canvas_height = 0
        
        # Augmentation options
        self.aug_options = {
            'flip_horizontal': tk.BooleanVar(value=False),
            'flip_vertical': tk.BooleanVar(value=False),
            'rotate': tk.BooleanVar(value=False),
            'brightness': tk.BooleanVar(value=False),
            'blur': tk.BooleanVar(value=False),
            'noise': tk.BooleanVar(value=False)
        }
        self.use_random_mask = tk.BooleanVar(value=True)
        self.save_augmentations = tk.BooleanVar(value=False)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side (2/3 width)
        left_frame = ttk.Frame(main_frame)
        left_frame.place(relx=0, rely=0, relwidth=0.67, relheight=1)
        
        # Right side (1/3 width)
        right_frame = ttk.Frame(main_frame)
        right_frame.place(relx=0.67, rely=0, relwidth=0.33, relheight=1)
        
        # Left top: Image canvas (2/3 height)
        self.setup_image_canvas(left_frame)
        
        # Left bottom: Browse options (1/3 height)
        self.setup_browse_section(left_frame)
        
        # Right top: Augmentation options (2/3 height)
        self.setup_augmentation_section(right_frame)
        
        # Right bottom: Download section (1/3 height)
        self.setup_download_section(right_frame)
        
    def setup_image_canvas(self, parent):
        """Setup the image display canvas"""
        canvas_frame = ttk.LabelFrame(parent, text="Image Viewer", padding=5)
        canvas_frame.place(relx=0, rely=0, relwidth=1, relheight=0.67)
        
        # Canvas for image display
        self.canvas = tk.Canvas(canvas_frame, bg='gray', cursor='cross')
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        
        # Bind mouse events for drawing
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        
        # Bind mouse events for panning with right-click
        self.canvas.bind('<ButtonPress-3>', self.on_pan_start)  # Right mouse
        self.canvas.bind('<B3-Motion>', self.on_pan_move)
        
        # Reset cursor when done
        self.canvas.bind('<ButtonRelease-2>', lambda e: self.on_pan_end())
        self.canvas.bind('<ButtonRelease-3>', lambda e: self.on_pan_end())
        
        # Bind mouse wheel for zooming
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        self.canvas.bind('<Button-4>', self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind('<Button-5>', self.on_mouse_wheel)  # Linux scroll down
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Delete>', lambda e: self.delete_selected_bbox())
        self.root.bind('<e>', lambda e: self.toggle_edit_mode())
        
        # Navigation buttons
        nav_frame = ttk.Frame(canvas_frame)
        nav_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="← Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        self.image_label = ttk.Label(nav_frame, text="No images loaded")
        self.image_label.pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_frame, text="Next →", command=self.next_image).pack(side=tk.RIGHT, padx=5)
        
        # Edit controls
        edit_frame = ttk.Frame(canvas_frame)
        edit_frame.pack(fill=tk.X, pady=2)
        
        self.edit_mode_btn = ttk.Button(edit_frame, text="Edit Mode (E)", command=self.toggle_edit_mode)
        self.edit_mode_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(edit_frame, text="Undo (Ctrl+Z)", command=self.undo).pack(side=tk.LEFT, padx=5)
        
        self.edit_status = ttk.Label(edit_frame, text="Draw Mode", foreground="green")
        self.edit_status.pack(side=tk.LEFT, padx=10)
        
        # Zoom controls
        zoom_frame = ttk.Frame(canvas_frame)
        zoom_frame.pack(fill=tk.X, pady=2)
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self.reset_zoom).pack(side=tk.LEFT, padx=5)
        
    def setup_browse_section(self, parent):
        """Setup the browse and load section"""
        browse_frame = ttk.LabelFrame(parent, text="Load Data", padding=10)
        browse_frame.place(relx=0, rely=0.67, relwidth=1, relheight=0.33)
        
        # Images and labels
        ttk.Label(browse_frame, text="YOLO Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Button(browse_frame, text="Browse Images", 
                   command=self.browse_images).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(browse_frame, text="Browse Labels", 
                   command=self.browse_labels).grid(row=0, column=2, padx=5, pady=5, sticky=tk.EW)
        
        self.dataset_status = ttk.Label(browse_frame, text="Status: No dataset loaded", 
                                       foreground="red")
        self.dataset_status.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Masks with class support
        ttk.Separator(browse_frame, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=3, 
                                                               sticky=tk.EW, pady=10)
        
        # Class management
        class_frame = ttk.Frame(browse_frame)
        class_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        ttk.Label(class_frame, text="Class:").pack(side=tk.LEFT, padx=5)
        self.class_combo = ttk.Combobox(class_frame, width=15, state='readonly')
        self.class_combo.pack(side=tk.LEFT, padx=5)
        self.class_combo.bind('<<ComboboxSelected>>', self.on_class_selected)
        
        ttk.Button(class_frame, text="Add Class", command=self.add_class).pack(side=tk.LEFT, padx=5)
        
        # Masks for current class
        ttk.Label(browse_frame, text="Masks for class:").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Button(browse_frame, text="Browse Masks", 
                   command=self.browse_masks).grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)
        
        self.mask_status = ttk.Label(browse_frame, text="Status: No masks loaded", 
                                     foreground="red")
        self.mask_status.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        browse_frame.columnconfigure(1, weight=1)
        browse_frame.columnconfigure(2, weight=1)
        
        # Initialize with class 0
        self.mask_classes[0] = {'name': 'Class 0', 'masks': [], 'augmented': []}
        self.update_class_combo()
        
    def setup_augmentation_section(self, parent):
        """Setup the augmentation options section"""
        aug_frame = ttk.LabelFrame(parent, text="Augmentation Options", padding=10)
        aug_frame.place(relx=0, rely=0, relwidth=1, relheight=0.67)
        
        ttk.Label(aug_frame, text="Masks are randomly selected", 
                 font=('Arial', 9, 'italic')).pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(aug_frame, text="Apply augmentations to masks:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        # Augmentation checkboxes
        ttk.Checkbutton(aug_frame, text="Horizontal Flip", 
                       variable=self.aug_options['flip_horizontal']).pack(anchor=tk.W, pady=3)
        
        ttk.Checkbutton(aug_frame, text="Vertical Flip", 
                       variable=self.aug_options['flip_vertical']).pack(anchor=tk.W, pady=3)
        
        ttk.Checkbutton(aug_frame, text="Random Rotation (±30°)", 
                       variable=self.aug_options['rotate']).pack(anchor=tk.W, pady=3)
        
        ttk.Checkbutton(aug_frame, text="Brightness Adjustment", 
                       variable=self.aug_options['brightness']).pack(anchor=tk.W, pady=3)
        
        ttk.Checkbutton(aug_frame, text="Gaussian Blur", 
                       variable=self.aug_options['blur']).pack(anchor=tk.W, pady=3)
        
        ttk.Checkbutton(aug_frame, text="Add Noise", 
                       variable=self.aug_options['noise']).pack(anchor=tk.W, pady=3)
        
        ttk.Separator(aug_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(aug_frame, text="Save Augmented Masks", 
                       variable=self.save_augmentations).pack(anchor=tk.W, pady=5)
        
        ttk.Button(aug_frame, text="Generate Augmentations", 
                  command=self.generate_augmentations).pack(fill=tk.X, pady=5, padx=10)
        
        self.aug_status = ttk.Label(aug_frame, text="", foreground="blue")
        self.aug_status.pack(pady=5)
        
        ttk.Separator(aug_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Instructions
        instructions = ttk.Label(aug_frame, text="Instructions:", 
                                font=('Arial', 10, 'bold'))
        instructions.pack(anchor=tk.W, pady=(5, 5))
        
        instr_text = """1. Load images, labels, and masks
2. Select mask or use random
3. Draw bounding box on image
4. Mask pasted at box center
5. Export when done"""
        
        ttk.Label(aug_frame, text=instr_text, justify=tk.LEFT, font=('Arial', 8)).pack(anchor=tk.W, padx=10)
        
    def setup_download_section(self, parent):
        """Setup the download/export section"""
        download_frame = ttk.LabelFrame(parent, text="Export Dataset", padding=10)
        download_frame.place(relx=0, rely=0.67, relwidth=1, relheight=0.33)
        
        ttk.Label(download_frame, text="Export annotated dataset\nwith new objects:", 
                 justify=tk.CENTER).pack(pady=10)
        
        ttk.Button(download_frame, text="Export Dataset", 
                  command=self.export_dataset, 
                  style='Accent.TButton').pack(pady=10, fill=tk.X, padx=20)
        
        self.export_status = ttk.Label(download_frame, text="", foreground="blue")
        self.export_status.pack(pady=5)
        
    def browse_images(self):
        """Browse and load YOLO images"""
        files = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not files:
            return
        
        self.image_paths = sorted(list(set(files)))  # Remove duplicates and sort
        self.images = []
        
        for img_path in self.image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                self.images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        self.current_image_idx = 0
        self.reset_zoom()
        self.update_dataset_status()
        self.display_current_image()
        
    def browse_labels(self):
        """Browse and load YOLO labels"""
        files = filedialog.askopenfilenames(
            title="Select Label Files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not files:
            return
        
        self.label_paths = sorted(list(set(files)))  # Remove duplicates and sort
        self.labels = []
        
        for label_path in self.label_paths:
            try:
                with open(label_path, 'r') as f:
                    labels = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            labels.append([float(x) for x in parts])
                    self.labels.append(labels)
            except Exception as e:
                print(f"Error loading {label_path}: {e}")
                self.labels.append([])
        
        self.update_dataset_status()
        
    def browse_masks(self):
        """Browse and load mask images for current class"""
        files = filedialog.askopenfilenames(
            title="Select Mask Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        
        if not files:
            return
        
        mask_paths = sorted(list(set(files)))  # Remove duplicates and sort
        masks = []
        
        for mask_path in mask_paths:
            try:
                mask = Image.open(mask_path).convert('RGBA')
                masks.append(mask)
            except Exception as e:
                print(f"Error loading {mask_path}: {e}")
        
        # Store masks for current class
        if self.current_class not in self.mask_classes:
            self.mask_classes[self.current_class] = {'name': f'Class {self.current_class}', 'masks': [], 'augmented': []}
        
        self.mask_classes[self.current_class]['masks'].extend(masks)
        
        # Update legacy references for compatibility
        self.mask_images = self.mask_classes[self.current_class]['masks']
        self.augmented_masks = self.mask_classes[self.current_class]['augmented']
        
        self.update_mask_status()
    
    def update_dataset_status(self):
        """Update dataset status label"""
        img_count = len(self.images)
        lbl_count = len(self.labels)
        
        if img_count > 0 and lbl_count > 0:
            if img_count == lbl_count:
                self.dataset_status.config(
                    text=f"Status: {img_count} images, {lbl_count} labels loaded ✓",
                    foreground="green"
                )
            else:
                self.dataset_status.config(
                    text=f"⚠ Warning: {img_count} images, {lbl_count} labels (mismatch!)",
                    foreground="orange"
                )
        elif img_count > 0:
            self.dataset_status.config(
                text=f"Status: {img_count} images loaded (no labels)",
                foreground="orange"
            )
        else:
            self.dataset_status.config(
                text="Status: No dataset loaded",
                foreground="red"
            )
    
    def update_mask_status(self):
        """Update mask status label"""
        if self.current_class in self.mask_classes:
            mask_count = len(self.mask_classes[self.current_class]['masks'])
            aug_count = len(self.mask_classes[self.current_class]['augmented'])
            
            if mask_count > 0:
                total = mask_count + aug_count
                self.mask_status.config(
                    text=f"Class {self.current_class}: {mask_count} masks, {aug_count} augmented (total: {total}) ✓",
                    foreground="green"
                )
            else:
                self.mask_status.config(
                    text=f"Class {self.current_class}: No masks loaded",
                    foreground="red"
                )
        else:
            self.mask_status.config(
                text="Status: No masks loaded",
                foreground="red"
            )
    
    def add_class(self):
        """Add a new mask class"""
        # Find next available class ID
        if self.mask_classes:
            new_class_id = max(self.mask_classes.keys()) + 1
        else:
            new_class_id = 0
        
        # Prompt for class name
        class_name = tk.simpledialog.askstring("Add Class", f"Enter name for Class {new_class_id}:", 
                                               initialvalue=f"Class {new_class_id}")
        if not class_name:
            return
        
        self.mask_classes[new_class_id] = {'name': class_name, 'masks': [], 'augmented': []}
        self.current_class = new_class_id
        self.update_class_combo()
        self.update_mask_status()
    
    def update_class_combo(self):
        """Update the class combobox"""
        values = [f"{cid}: {self.mask_classes[cid]['name']}" for cid in sorted(self.mask_classes.keys())]
        self.class_combo['values'] = values
        if values:
            # Set current class
            for i, cid in enumerate(sorted(self.mask_classes.keys())):
                if cid == self.current_class:
                    self.class_combo.current(i)
                    break
    
    def on_class_selected(self, event):
        """Handle class selection change"""
        selection = self.class_combo.get()
        if selection:
            class_id = int(selection.split(':')[0])
            self.current_class = class_id
            # Update references
            self.mask_images = self.mask_classes[self.current_class]['masks']
            self.augmented_masks = self.mask_classes[self.current_class]['augmented']
            self.update_mask_status()
    
    def display_current_image(self):
        """Display the current image on canvas"""
        if not self.images:
            return
        
        img = self.images[self.current_image_idx].copy()
        
        # Draw existing bounding boxes on original image
        if self.current_image_idx < len(self.labels):
            img = self.draw_yolo_boxes(img, self.labels[self.current_image_idx])
        
        # Store original dimensions
        self.original_img_width = img.width
        self.original_img_height = img.height
        
        # Apply zoom to the image
        zoomed_width = int(img.width * self.zoom_level)
        zoomed_height = int(img.height * self.zoom_level)
        img = img.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, use defaults
            canvas_width = zoomed_width
            canvas_height = zoomed_height
        
        # Calculate new offset to maintain the center point
        if hasattr(self, 'last_display_center_x') and hasattr(self, 'last_display_center_y'):
            # We have previous center point, maintain it
            center_x = self.last_display_center_x
            center_y = self.last_display_center_y
            
            # Calculate new offset based on center point and new image size
            self.canvas_offset_x = center_x - zoomed_width / 2
            self.canvas_offset_y = center_y - zoomed_height / 2
        else:
            # First display, center the image
            self.canvas_offset_x = max(0, (canvas_width - zoomed_width) // 2)
            self.canvas_offset_y = max(0, (canvas_height - zoomed_height) // 2)
        
        # Clamp offset to ensure image stays visible
        if self.canvas_offset_x > 0:
            self.canvas_offset_x = 0
        elif self.canvas_offset_x + zoomed_width < canvas_width:
            self.canvas_offset_x = canvas_width - zoomed_width
        
        if self.canvas_offset_y > 0:
            self.canvas_offset_y = 0
        elif self.canvas_offset_y + zoomed_height < canvas_height:
            self.canvas_offset_y = canvas_height - zoomed_height
        
        # Store the center point for next update
        self.last_display_center_x = self.canvas_offset_x + zoomed_width / 2
        self.last_display_center_y = self.canvas_offset_y + zoomed_height / 2
        
        # Store the scale factor (pixels on screen per pixel in original image)
        self.canvas_scale = self.zoom_level
        
        self.display_image = img
        self.photo_image = ImageTk.PhotoImage(img)
        
        self.canvas.delete('all')
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, anchor=tk.NW, image=self.photo_image)
        
        # Update labels
        total = len(self.images)
        filename = Path(self.image_paths[self.current_image_idx]).name
        self.image_label.config(text=f"Image {self.current_image_idx + 1}/{total}: {filename}")
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
    
    def draw_yolo_boxes(self, img, labels):
        """Draw YOLO bounding boxes on image"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Color palette for different classes
        colors = ['green', 'blue', 'red', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        
        for idx, label in enumerate(labels):
            class_id, x_center, y_center, w, h = label
            
            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)
            
            # Select color based on class
            color = colors[int(class_id) % len(colors)]
            
            # Highlight selected bbox
            if self.editing_mode and idx == self.selected_bbox_idx:
                # Draw thicker border for selected bbox
                draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
                # Draw resize handles (small squares at corners) - size based on bbox dimensions
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                # Make handle size proportional to bbox size, but within reasonable limits
                handle_size = max(2, min(12, int(min(bbox_width, bbox_height) * 0.05))) / self.canvas_scale
                for corner_x, corner_y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    draw.rectangle(
                        [corner_x - handle_size, corner_y - handle_size, 
                        corner_x + handle_size, corner_y + handle_size],
                        outline='white', fill=color, width=2
                    )
            else:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            
            # Draw class label
            label_text = f"C{int(class_id)}"
            draw.text((x1, max(0, y1-15)), label_text, fill=color)
        
        return img
    
    def prev_image(self):
        """Navigate to previous image"""
        if self.images and self.current_image_idx > 0:
            self.current_image_idx -= 1
            # Don't reset zoom when navigating images, keep current zoom level
            self.display_current_image()
    
    def next_image(self):
        """Navigate to next image"""
        if self.images and self.current_image_idx < len(self.images) - 1:
            self.current_image_idx += 1
            # Don't reset zoom when navigating images, keep current zoom level
            self.display_current_image()
    
    def zoom_in(self):
        """Zoom in on the image"""
        old_zoom = self.zoom_level
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        if old_zoom != self.zoom_level:
            self.display_current_image()
    
    def zoom_out(self):
        """Zoom out on the image"""
        old_zoom = self.zoom_level
        self.zoom_level = max(self.zoom_level / 1.2, 0.5)
        if old_zoom != self.zoom_level:
            self.display_current_image()
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_level = 1.0
        self.display_current_image()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if event.num == 5 or event.delta < 0:
            self.zoom_out()
        elif event.num == 4 or event.delta > 0:
            self.zoom_in()
    
    def on_pan_start(self, event):
        """Start panning the image"""
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor='fleur')
    
    def on_pan_move(self, event):
        """Pan the image"""
        if not self.panning:
            return
        
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        self.canvas_offset_x += dx
        self.canvas_offset_y += dy
        
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
        # Calculate and store the new center point
        if self.display_image:
            zoomed_width = self.display_image.width
            zoomed_height = self.display_image.height
            self.last_display_center_x = self.canvas_offset_x + zoomed_width / 2
            self.last_display_center_y = self.canvas_offset_y + zoomed_height / 2
        
        # Redraw image at new position
        self.canvas.delete('all')
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, anchor=tk.NW, image=self.photo_image)
    
    def on_pan_end(self, event=None):
        """End panning"""
        self.panning = False
        self.canvas.config(cursor='cross')
        
        # Make sure center point is updated after panning
        if self.display_image:
            zoomed_width = self.display_image.width
            zoomed_height = self.display_image.height
            self.last_display_center_x = self.canvas_offset_x + zoomed_width / 2
            self.last_display_center_y = self.canvas_offset_y + zoomed_height / 2
    
    def toggle_edit_mode(self):
        """Toggle between draw and edit mode"""
        self.editing_mode = not self.editing_mode
        
        if self.editing_mode:
            self.edit_status.config(text="Edit Mode (Move/Resize boxes)", foreground="orange")
            self.edit_mode_btn.config(text="Draw Mode (E)")
            self.selected_bbox_idx = None
        else:
            self.edit_status.config(text="Draw Mode", foreground="green")
            self.edit_mode_btn.config(text="Edit Mode (E)")
            self.selected_bbox_idx = None
        
        self.display_current_image()
    
    def select_bbox_at_position(self, canvas_x, canvas_y):
        """Select a bounding box at the given canvas position"""
        if self.current_image_idx >= len(self.labels):
            return
        
        # Convert canvas coordinates to image coordinates
        img_x = (canvas_x - self.canvas_offset_x) / self.canvas_scale
        img_y = (canvas_y - self.canvas_offset_y) / self.canvas_scale
        
        labels = self.labels[self.current_image_idx]
        img_width = self.original_img_width
        img_height = self.original_img_height
        
        # Check each bbox from back to front (last drawn on top)
        for idx in range(len(labels) - 1, -1, -1):
            label = labels[idx]
            class_id, x_center, y_center, w, h = label
            
            # Convert to pixel coordinates
            x1 = (x_center - w/2) * img_width
            y1 = (y_center - h/2) * img_height
            x2 = (x_center + w/2) * img_width
            y2 = (y_center + h/2) * img_height
            
            # Check if click is inside bbox or on resize handle
            handle_size = 8 / self.canvas_scale  # Adjust handle size based on zoom
            
            # Check resize handles first
            if (abs(img_x - x1) <= handle_size and abs(img_y - y1) <= handle_size):
                self.selected_bbox_idx = idx
                self.resize_corner = 'top-left'
                self.dragging_resize = True
                self.drag_start_x = canvas_x
                self.drag_start_y = canvas_y
                return
            elif (abs(img_x - x2) <= handle_size and abs(img_y - y1) <= handle_size):
                self.selected_bbox_idx = idx
                self.resize_corner = 'top-right'
                self.dragging_resize = True
                self.drag_start_x = canvas_x
                self.drag_start_y = canvas_y
                return
            elif (abs(img_x - x1) <= handle_size and abs(img_y - y2) <= handle_size):
                self.selected_bbox_idx = idx
                self.resize_corner = 'bottom-left'
                self.dragging_resize = True
                self.drag_start_x = canvas_x
                self.drag_start_y = canvas_y
                return
            elif (abs(img_x - x2) <= handle_size and abs(img_y - y2) <= handle_size):
                self.selected_bbox_idx = idx
                self.resize_corner = 'bottom-right'
                self.dragging_resize = True
                self.drag_start_x = canvas_x
                self.drag_start_y = canvas_y
                return
            # Check if click is inside bbox
            elif x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.selected_bbox_idx = idx
                self.dragging_bbox = True
                self.drag_start_x = canvas_x
                self.drag_start_y = canvas_y
                self.resize_corner = None
                return
        
        # No bbox found
        self.selected_bbox_idx = None
        self.resize_corner = None
        self.display_current_image()
    
    def delete_selected_bbox(self):
        """Delete the currently selected bounding box"""
        if not self.editing_mode or self.selected_bbox_idx is None:
            return
        
        if self.current_image_idx >= len(self.labels):
            return
        
        # Save state for undo
        self.save_undo_state()
        
        # Delete bbox
        del self.labels[self.current_image_idx][self.selected_bbox_idx]
        self.selected_bbox_idx = None
        self.resize_corner = None
        self.display_current_image()
    
    def save_undo_state(self):
        """Save current state to undo stack"""
        if self.current_image_idx >= len(self.images):
            return
        
        state = {
            'image_idx': self.current_image_idx,
            'image': self.images[self.current_image_idx].copy(),
            'labels': copy.deepcopy(self.labels[self.current_image_idx]) if self.current_image_idx < len(self.labels) else []
        }
        
        self.undo_stack.append(state)
        
        # Limit undo stack size
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
    
    def undo(self):
        """Undo the last action"""
        if not self.undo_stack:
            messagebox.showinfo("Undo", "Nothing to undo")
            return
        
        state = self.undo_stack.pop()
        
        # Restore state
        img_idx = state['image_idx']
        if img_idx < len(self.images):
            self.images[img_idx] = state['image']
            
            while len(self.labels) <= img_idx:
                self.labels.append([])
            
            self.labels[img_idx] = state['labels']
            
            if self.current_image_idx == img_idx:
                self.display_current_image()
    
    def on_mouse_down(self, event):
        """Handle mouse button press"""
        if not self.images:
            return
        
        # In edit mode, select bbox
        if self.editing_mode:
            self.select_bbox_at_position(event.x, event.y)
            return
        
        # In draw mode, check if we have masks
        if not self.mask_images:
            messagebox.showinfo("No Masks", "Please load masks first.")
            return
        
        # Single click to place mask at original size
        canvas_x = event.x
        canvas_y = event.y
        
        # Convert canvas coordinates to original image coordinates
        img_x = int((canvas_x - self.canvas_offset_x) / self.canvas_scale)
        img_y = int((canvas_y - self.canvas_offset_y) / self.canvas_scale)
        
        # Clamp to image bounds
        img_x = max(0, min(img_x, self.original_img_width))
        img_y = max(0, min(img_y, self.original_img_height))
        
        # Save state for undo
        self.save_undo_state()
        
        # Paste mask at clicked position
        self.paste_mask_at_position(img_x, img_y)
        self.display_current_image()

    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        # Handle bbox dragging in edit mode
        if self.editing_mode:
            if self.dragging_bbox and self.selected_bbox_idx is not None:
                dx = event.x - self.drag_start_x
                dy = event.y - self.drag_start_y
                
                # Convert to original image coordinates
                dx_img = dx / self.canvas_scale
                dy_img = dy / self.canvas_scale
                
                # Update bbox position
                if self.current_image_idx < len(self.labels):
                    labels = self.labels[self.current_image_idx]
                    if self.selected_bbox_idx < len(labels):
                        label = labels[self.selected_bbox_idx]
                        img_width = self.original_img_width
                        img_height = self.original_img_height
                        
                        # Update center position
                        new_x = label[1] + (dx_img / img_width)
                        new_y = label[2] + (dy_img / img_height)
                        
                        # Clamp to image bounds
                        w, h = label[3], label[4]
                        new_x = max(w/2, min(new_x, 1 - w/2))
                        new_y = max(h/2, min(new_y, 1 - h/2))
                        
                        labels[self.selected_bbox_idx][1] = new_x
                        labels[self.selected_bbox_idx][2] = new_y
                
                self.drag_start_x = event.x
                self.drag_start_y = event.y
                self.display_current_image()
                return
            
            elif self.dragging_resize and self.selected_bbox_idx is not None and self.resize_corner:
                # Handle bbox resizing
                dx = event.x - self.drag_start_x
                dy = event.y - self.drag_start_y
                
                # Convert to original image coordinates
                dx_img = dx / self.canvas_scale
                dy_img = dy / self.canvas_scale
                
                # Update bbox size
                if self.current_image_idx < len(self.labels):
                    labels = self.labels[self.current_image_idx]
                    if self.selected_bbox_idx < len(labels):
                        label = labels[self.selected_bbox_idx]
                        img_width = self.original_img_width
                        img_height = self.original_img_height
                        
                        # Convert to pixel coordinates
                        x_center = label[1] * img_width
                        y_center = label[2] * img_height
                        w = label[3] * img_width
                        h = label[4] * img_height
                        
                        # Get current corners
                        x1 = x_center - w/2
                        y1 = y_center - h/2
                        x2 = x_center + w/2
                        y2 = y_center + h/2
                        
                        # Adjust based on which corner is being dragged
                        if self.resize_corner == 'top-left':
                            x1 += dx_img
                            y1 += dy_img
                        elif self.resize_corner == 'top-right':
                            x2 += dx_img
                            y1 += dy_img
                        elif self.resize_corner == 'bottom-left':
                            x1 += dx_img
                            y2 += dy_img
                        elif self.resize_corner == 'bottom-right':
                            x2 += dx_img
                            y2 += dy_img
                        
                        # Ensure minimum size
                        min_size = 5
                        if abs(x2 - x1) < min_size:
                            if self.resize_corner in ['top-left', 'bottom-left']:
                                x1 = x2 - min_size
                            else:
                                x2 = x1 + min_size
                        
                        if abs(y2 - y1) < min_size:
                            if self.resize_corner in ['top-left', 'top-right']:
                                y1 = y2 - min_size
                            else:
                                y2 = y1 + min_size
                        
                        # Clamp to image bounds
                        x1 = max(0, min(x1, img_width))
                        x2 = max(0, min(x2, img_width))
                        y1 = max(0, min(y1, img_height))
                        y2 = max(0, min(y2, img_height))
                        
                        # Convert back to YOLO format
                        new_x_center = (x1 + x2) / 2 / img_width
                        new_y_center = (y1 + y2) / 2 / img_height
                        new_w = (x2 - x1) / img_width
                        new_h = (y2 - y1) / img_height
                        
                        # Update label
                        labels[self.selected_bbox_idx][1] = new_x_center
                        labels[self.selected_bbox_idx][2] = new_y_center
                        labels[self.selected_bbox_idx][3] = new_w
                        labels[self.selected_bbox_idx][4] = new_h
                
                self.drag_start_x = event.x
                self.drag_start_y = event.y
                self.display_current_image()
                return
        
        # No drawing rectangle in draw mode anymore

    def on_mouse_up(self, event):
        """Handle mouse button release"""
        # Stop dragging/resizing in edit mode
        if self.editing_mode:
            if self.dragging_bbox or self.dragging_resize:
                # Save state for undo after finishing drag/resize
                self.save_undo_state()
                self.dragging_bbox = False
                self.dragging_resize = False
            return
        
        # No drawing rectangle to clean up anymore

    def paste_mask_at_position(self, center_x, center_y):
        """Paste random mask onto the image at specified position with original size"""
        # Get masks for current class
        if self.current_class not in self.mask_classes:
            messagebox.showwarning("No Masks", "Please load masks for this class first.")
            return
        
        masks = self.mask_classes[self.current_class]['masks']
        augmented = self.mask_classes[self.current_class]['augmented']
        
        # Use random mask from original masks + augmented masks
        all_masks = masks + augmented
        if not all_masks:
            messagebox.showwarning("No Masks", "No masks available for this class.")
            return
        
        mask = random.choice(all_masks).copy()
        
        # Apply real-time augmentations (on top of pre-generated ones)
        mask = self.apply_augmentations(mask)
        
        # Use original mask size
        mask_width, mask_height = mask.size
        
        # Calculate paste position (top-left corner, centered on click)
        paste_x = center_x - mask_width // 2
        paste_y = center_y - mask_height // 2
        
        # Clamp to image bounds
        paste_x = max(0, min(paste_x, self.original_img_width - mask_width))
        paste_y = max(0, min(paste_y, self.original_img_height - mask_height))
        
        # Recalculate actual center after clamping
        actual_center_x = paste_x + mask_width // 2
        actual_center_y = paste_y + mask_height // 2
        
        # Paste mask onto current image
        img = self.images[self.current_image_idx]
        img.paste(mask, (paste_x, paste_y), mask)
        
        # Add new YOLO annotation with current class
        img_width, img_height = img.size
        x_center = actual_center_x / img_width
        y_center = actual_center_y / img_height
        w = mask_width / img_width
        h = mask_height / img_height
        
        # Use current class ID
        new_annotation = [self.current_class, x_center, y_center, w, h]
        
        # Ensure labels list exists for this image
        while len(self.labels) <= self.current_image_idx:
            self.labels.append([])
        
        self.labels[self.current_image_idx].append(new_annotation)
    
    def generate_augmentations(self):
        """Generate augmented versions of masks and optionally save them"""
        if self.current_class not in self.mask_classes:
            messagebox.showwarning("No Masks", "Please load masks for this class first.")
            return
        
        masks = self.mask_classes[self.current_class]['masks']
        if not masks:
            messagebox.showwarning("No Masks", "Please load masks for this class first.")
            return
        
        augmented = []
        
        # Create augmenters based on selected options
        augmenters = []
        
        if self.aug_options['flip_horizontal'].get():
            augmenters.append(iaa.Fliplr(1.0))
        
        if self.aug_options['flip_vertical'].get():
            augmenters.append(iaa.Flipud(1.0))
        
        if self.aug_options['rotate'].get():
            for angle in [-30, -15, 15, 30]:
                augmenters.append(iaa.Rotate(angle))
        
        if self.aug_options['brightness'].get():
            for mult in [0.7, 1.3]:
                augmenters.append(iaa.Multiply(mult))
        
        if self.aug_options['blur'].get():
            augmenters.append(iaa.GaussianBlur(sigma=1.0))
        
        if self.aug_options['noise'].get():
            augmenters.append(iaa.AdditiveGaussianNoise(scale=0.05*255))
        
        if not augmenters:
            messagebox.showinfo("No Augmentations", "Please select at least one augmentation option.")
            return
        
        # Apply each augmentation to each mask
        for mask in masks:
            mask_np = np.array(mask)
            for aug in augmenters:
                augmented_np = aug(image=mask_np)
                augmented_mask = Image.fromarray(augmented_np)
                augmented.append(augmented_mask)
        
        # Store in class
        self.mask_classes[self.current_class]['augmented'] = augmented
        
        # Update references
        self.augmented_masks = augmented
        
        # Save augmented masks if option is selected
        if self.save_augmentations.get():
            save_dir = filedialog.askdirectory(title="Select Directory to Save Augmented Masks")
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(exist_ok=True)
                
                for i, aug_mask in enumerate(augmented):
                    aug_mask.save(save_path / f"class{self.current_class}_augmented_{i:04d}.png")
                
                messagebox.showinfo("Success", f"Saved {len(augmented)} augmented masks to {save_dir}")
        
        self.aug_status.config(text=f"✓ Generated {len(augmented)} augmented masks for class {self.current_class}")
        self.update_mask_status()
    
    def apply_augmentations(self, mask):
        """Apply selected augmentations to mask"""
        # Convert PIL to numpy for imgaug
        mask_np = np.array(mask)
        
        augmenters = []
        
        if self.aug_options['flip_horizontal'].get():
            augmenters.append(iaa.Fliplr(0.5))
        
        if self.aug_options['flip_vertical'].get():
            augmenters.append(iaa.Flipud(0.5))
        
        if self.aug_options['rotate'].get():
            augmenters.append(iaa.Rotate((-30, 30)))
        
        if self.aug_options['brightness'].get():
            augmenters.append(iaa.Multiply((0.7, 1.3)))
        
        if self.aug_options['blur'].get():
            augmenters.append(iaa.GaussianBlur(sigma=(0, 1.0)))
        
        if self.aug_options['noise'].get():
            augmenters.append(iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)))
        
        if augmenters:
            seq = iaa.Sequential(augmenters)
            mask_np = seq(image=mask_np)
        
        return Image.fromarray(mask_np)
    
    def export_dataset(self):
        """Export the modified dataset with new annotations"""
        if not self.images:
            messagebox.showwarning("No Data", "Please load images first.")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        output_path = Path(output_dir)
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        try:
            for i, (img, img_path) in enumerate(zip(self.images, self.image_paths)):
                # Save image
                filename = Path(img_path).stem
                img.save(images_dir / f"{filename}.png")
                
                # Save labels
                if i < len(self.labels):
                    label_file = labels_dir / f"{filename}.txt"
                    with open(label_file, 'w') as f:
                        for label in self.labels[i]:
                            f.write(' '.join([str(x) for x in label]) + '\n')
            
            self.export_status.config(text=f"✓ Exported to {output_dir}")
            messagebox.showinfo("Success", f"Dataset exported successfully to:\n{output_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting dataset:\n{str(e)}")


def main():
    root = tk.Tk()
    app = YOLOMaskAnnotator(root)
    root.mainloop()


if __name__ == "__main__":
    main()