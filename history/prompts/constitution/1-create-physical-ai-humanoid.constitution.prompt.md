---
id: 1
title: Create Physical AI Humanoid Constitution
stage: constitution
date: 2025-12-07
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution, textbook, ai, robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - null
---

## Prompt

```text
Project Overview:

Create an AI-native technical textbook titled Physical AI & Humanoid Robotics for a capstone quarter-long course focused on embodied intelligence and AI systems in the physical world. The book teaches students to bridge digital AI with physical robotics, covering design, simulation, and deployment of humanoid robots using ROS 2, Gazebo, NVIDIA Isaac, Unity, and Vision-Language-Action (VLA) models. It must be built with Docusaurus, include an embedded Retrieval-Augmented Generation (RAG) chatbot, and support bonus features like user signup/signin, per-chapter personalization, and Urdu translation.



Core Principles:



Embodied Intelligence Focus



Practical Reproducibility



Accessibility & Inclusivity



AI-Native Interactivity



Ethical Robotics



Quality Standards:



Clear learning objectives per chapter



High technical accuracy with validated code examples



Clear writing style (grade 8–10)



Visual aids (diagrams, images, code blocks)



Integration of interactive features (RAG chatbot, personalization, translations)



Original content with citations and no plagiarism



Source Requirements:



60%+ from official documentation or peer-reviewed sources



Diversity of sources (open-source repos, papers, tutorials)



5–10 references per chapter



All sources fact-checked and reproducible



Constraints:



200–300 page textbook



Written in Docusaurus Markdown



Built for deployment to GitHub Pages/Vercel



Technical stack: Ubuntu 22.04, ROS 2 Humble/Iron, Gazebo Harmonic, NVIDIA Isaac Sim, Unity 2022+



Audience: intermediate AI/Python students



Structured into 4 modules across 13 weeks



Non-goals: deep RL theory, non-NVIDIA hardware deep dives



Success Criteria:



Complete coverage of the modules and capstone



Usable and reproducible exercises



80%+ runnable code examples



Accessibility compliance



Functional interactive features



Students can build/simulate a basic humanoid robot



Generate the constitution.md file with the above specifications, focusing strictly on the reproducibility, scope, constraints, principles, standards, and success metrics of the project.
```

## Response snapshot

Created the Physical AI & Humanoid Robotics Constitution with six core principles: Embodied Intelligence Focus, Practical Reproducibility, Accessibility & Inclusivity, AI-Native Interactivity, Ethical Robotics, and Technical Excellence. The constitution also includes Quality Standards, Source Requirements, Constraints, and Success Criteria sections that align with the project specifications. The file was written to .specify/memory/constitution.md with version 1.0.0, ratification date 2025-12-07, and last amended date 2025-12-07.

## Outcome

- ✅ Impact: Created initial project constitution for Physical AI & Humanoid Robotics textbook
- 🧪 Tests: N/A (documentation task)
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Plan and implement textbook features based on constitution
- 🧠 Reflection: Constitution provides clear guidance for textbook development with focus on embodied intelligence and practical reproducibility

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Create feature specifications based on constitution