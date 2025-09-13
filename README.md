# NocsFlow via Only one Image

## Abstract
Accurate pose estimation and tracking are crucial for physical AI systems such as robots; however, existing methods often fail to quantify the uncertainty of their predictions. To address this limitation, we propose **NocsFlow**, a novel pose estimation framework. Our approach leverages NOCS maps as the pose representation and employs Flow Matching to generate the final NOCS map. Subsequently, pose estimation is obtained through point cloud registration using Umelaya. By performing multiple inferences, our method is able to capture uncertainty across different dimensions, thereby enabling robust uncertainty quantification for symmetric object pose estimation.

---
![workflow](images/workflow.png)
## 1. Introduction
Estimating the six degrees of freedom (6D) rigid-body transformation between an object and a camera—commonly known as *object pose estimation*—is a fundamental task in robotic manipulation and augmented reality. Despite remarkable progress in recent years, existing approaches still face key limitations that hinder their deployment in real-world scenarios.

First, most methods operate at the **instance level**, making it difficult to generalize to unseen objects. Second, they often rely heavily on **prior information**, such as object masks or bounding boxes, which restricts their applicability. These issues largely stem from the dependence of model inputs on extensive reference information. In practical deployment, however, a system must be able to handle diverse and previously unknown objects without extensive preparation.

Moreover, many current models require explicit 3D models or CAD templates as input, which not only increases data preparation costs but also limits flexibility in dynamic or cluttered environments. To address these challenges, researchers have explored two promising directions:

- **Flow matching**, a recent generative technique for learning continuous transformations, provides a powerful way to model the dense correspondence between 2D image observations and 3D object geometry, thereby improving both pose estimation accuracy and robustness to occlusion.
- **Normalized Object Coordinate Space (NOCS)** representations map every object pixel to a normalized canonical space, enabling category-level generalization and metric-scale 6D pose recovery without the need for exact 3D models.

Combining these strengths offers a compelling path forward for practical industrial applications.

In this work, we introduce **NocsFM**, a unified framework that leverages **DINOv3** features for reliable cross-view matching and integrates flow matching with NOCS-based learning to directly infer object pose and masks from a single RGB reference image. Our method automatically tracks the object over time using the predicted mask, eliminating the need for manual re-initialization or object-specific CAD models.

**Main contributions:**
- **Unified framework for pose estimation and tracking.** Requires only one reference image to continuously track objects and estimate their 6D poses in real time.
- **First use of flow matching on NOCS.** Pioneers the application of flow matching within the NOCS representation, enabling robust and category-level 6D pose estimation directly from RGB images.
- **Joint mask prediction and pose estimation.** Couples mask prediction with pose estimation, enabling fully automated object tracking and reducing the dependency on external detectors or handcrafted priors.

By unifying feature matching, flow-based geometric reasoning, and NOCS-based 3D representation, **NocsFM** advances object pose estimation towards truly generalizable and deployment-ready solutions for robotics, augmented reality, and industrial automation.

---

## 2. Related Work

### 2.1 6D Pose Estimation
6D object pose estimation aims to recover an object's 3D translation and rotation from one or more RGB/RGB-D images, serving as a fundamental technique for robotic manipulation, augmented reality, and industrial automation. Existing methods can roughly be divided into two categories:

- **Geometry- and template-based methods** rely on known CAD models and geometric algorithms such as keypoint matching, edge alignment, or iterative closest point (ICP). These approaches are sensitive to lighting changes, occlusions, cluttered backgrounds, and unknown objects.
- **Learning-based methods** directly regress poses or detect keypoints using CNN or Transformer architectures. While they improve robustness and accuracy, they typically require large-scale annotated data, instance-level 3D models, or multi-view priors.

Recently, approaches integrating large vision models and generative modeling (e.g., **FoundationPose**, **DiffusionNOCS**) have emerged to improve generalization and data efficiency.

### 2.2 FoundationPose
FoundationPose leverages large-scale pre-trained vision models as general-purpose feature extractors and adapts to downstream pose estimation tasks with limited task-specific data. However, it still typically requires known 3D models or accurate category labels, and may fail in dynamic scenes with fast motion or severe occlusion.

### 2.3 DiffusionNOCS
DiffusionNOCS combines diffusion models with NOCS to generate normalized object coordinate fields through reverse diffusion. While effective in modeling shape variations, it is computationally expensive, time-consuming, and sensitive to noise.

### 2.4 Our Advantages
**NocsFM** overcomes these shortcomings:
- Requires only a single reference image to continuously track and estimate an object’s 6D pose in open-world scenarios.
- Uses **flow matching** for one-shot dense correspondence estimation, reducing inference time compared to diffusion-based methods.
- Couples mask prediction with pose estimation, enhancing tracking stability and automation.

In summary, NocsFM inherits the generalization of FoundationPose and the category-level representation power of NOCS-based methods, while overcoming their reliance on templates, inference latency, and multi-stage complexity.

---

## 3. Method
**NocsFM** performs model-free 6D pose estimation and object tracking from RGB input in a single, unified loop. Given a clean reference image of the target object and the current RGB frame, the pipeline includes three stages:

1. **DINOv3 matching**
2. **Flow matching for NOCS estimation**
3. **Mask-based tracking**

### 3.1 DINOv3 Matching
A clean reference image and the current RGB frame are processed using DINOv3 to localize the target object. Dense feature embeddings from both images are matched to identify the object region, which is cropped and passed to the flow matching module. This step is category-agnostic and does not require predefined templates.

### 3.2 Flow Matching for NOCS Estimation
The cropped object patch is fed into a flow matching network to estimate dense correspondence between object surface and NOCS. Unlike diffusion-based approaches, flow matching performs the transformation in a single pass, producing:
- A dense **NOCS map** for 6D pose recovery.
- A **foreground mask** delineating the object.

This one-shot process ensures precision and real-time efficiency.

### 3.3 Tracking with Mask Feedback
The predicted mask:
- Enables precise pose computation by isolating the object.
- Generates a new clean reference image, which is fed back into the next iteration of DINOv3 matching.

This feedback mechanism forms a closed-loop system for long-term, automated tracking without manual re-initialization.

### 3.4 Summary
By integrating DINOv3 visual features, one-shot flow matching for NOCS estimation, and mask-based tracking, **NocsFM** provides a simple yet effective solution for category-level 6D pose estimation. The pipeline operates in real time and requires only a single initial reference image.

---

## 4. Experimental Evaluation

### 4.1 Setup
Experiments were conducted on a workstation with:
- NVIDIA RTX 3090 GPU (24 GB memory)
- Intel Xeon 3.0 GHz CPU
- 128 GB RAM

The method is implemented in PyTorch, using DINOv3 for feature extraction and a customized flow matching network for NOCS prediction and pose estimation. Training and evaluation were performed on the **CAMERA** dataset.

### 4.2 Evaluation of DINOv3 Matching
DINOv3, trained with a reference-to-frame matching objective, demonstrated robust object localization across diverse conditions, confirming its effectiveness as the first stage of NocsFM.

### 4.3 Evaluation of Flow Matching for Pose Estimation
Flow matching accurately predicted dense NOCS maps and masks, enabling stable pose estimation without iterative sampling. This supports real-time operation when integrated into the full NocsFM pipeline.

---

## 5. Conclusion
We presented **NocsFM**, a unified framework for category-level 6D object pose estimation and tracking from RGB input. By combining DINOv3-based feature matching, one-shot flow matching for NOCS prediction, and a mask-driven feedback loop, NocsFM requires only a single reference image and no CAD models while operating in real time. Experiments on the CAMERA dataset demonstrate accurate pose estimation and robust long-term tracking, advancing 6D pose estimation toward practical, deployment-ready applications.
