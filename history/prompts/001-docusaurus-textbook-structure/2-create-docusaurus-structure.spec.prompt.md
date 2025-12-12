---
id: 2
title: Create Docusaurus Textbook Structure Spec
stage: spec
date: 2025-12-07
surface: agent
model: claude-sonnet-4-5-20250929
feature: docusaurus-textbook-structure
branch: 001-docusaurus-textbook-structure
user: user
command: /sp.specify
labels: [specification, docusaurus, textbook, ai, robotics]
links:
  spec: specs/001-docusaurus-textbook-structure/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/001-docusaurus-textbook-structure/spec.md
 - specs/001-docusaurus-textbook-structure/space.md
 - specs/001-docusaurus-textbook-structure/checklists/requirements.md
tests:
 - null
---

## Prompt

```text
Project Context: Create the spatial structure for the Physical AI & Humanoid Robotics Docusaurus-based textbook. The book supports a 13-week embodied AI robotics course using ROS 2, Gazebo, NVIDIA Isaac, Unity, and VLA models. Interactive features include RAG chatbot, Better Auth signup, personalization button, and Urdu translation button.

Space Definition:

Structure: Docusaurus site with sidebar by 4 modules and 13 weeks. ~250 total pages across chapters, intro, and appendices.

Artifacts:

docs/ for MDX chapters

src/components/ for chatbot, auth modal, personalization UI

static/ for images/videos

docusaurus.config.js for theme, i18n, plugins

Optional: agents/ and api/ folders for Claude Subagents and RAG endpoints

UX: Sidebar navigation, global search, persistent RAG chatbot widget, user dashboard after signup.

Content Structure: Each chapter includes objectives, theory, code examples, simulations, exercises, visuals.

Modular Breakdown:

Preface/Week 0

Module 1 (Weeks 1–5): 4 chapters—ROS 2 basics to URDF humanoid setup

Module 2 (Weeks 6–7): 2 chapters—Gazebo & Unity

Module 3 (Weeks 8–10): 3 chapters—NVIDIA Isaac

Module 4 (Weeks 11–13): 3 chapters—VLA integrations and capstone

Appendices: Hardware tiers, cloud alternatives, glossary, references

Constraints & Inheritance:

Follow constitution principles (reproducibility, clarity, accessibility). Use Docusaurus v3+. Ensure compatibility with Ubuntu 22.04/RTX GPU systems.

Output Expectations:

Generate specify/space.md including:

Site map (Mermaid diagram)

Folder structure

Chapter TOC with approximate word counts

UI wireframes (text-based)

Suggestions for two robotics-focused enhancements
```

## Response snapshot

Created the complete spatial structure for the Physical AI & Humanoid Robotics textbook with:
- Created branch 001-docusaurus-textbook-structure
- Generated comprehensive space.md with site map, folder structure, chapter TOC (~200k words), UI wireframes, and two robotics enhancements
- Created detailed specification with user stories, requirements, and success criteria
- Generated quality checklist to validate the specification

The space.md includes a Mermaid site map, detailed folder structure organized by modules and weeks, comprehensive chapter TOC with word counts, text-based UI wireframes, and two robotics-focused enhancement suggestions (Interactive 3D Robot Simulator and Hardware Abstraction Layer).

## Outcome

- ✅ Impact: Created complete spatial structure and specification for Docusaurus textbook
- 🧪 Tests: N/A (documentation task)
- 📁 Files: Created spec.md, space.md, and requirements checklist
- 🔁 Next prompts: Plan implementation based on specification
- 🧠 Reflection: Specification follows constitution principles with clear user scenarios and measurable success criteria

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): All validation checks passed
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Create implementation plan based on specification