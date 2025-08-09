# 🎵 Music Generator AI

Hi 🤙 This project is a SaaS application that generates original music using AI.  
It uses a **state-of-the-art music generation model** to create songs from simple text descriptions, custom lyrics, or style prompts.  

The platform supports **instrumental or vocal tracks**, offers a **community music feed** for sharing creations, and allows users to manage their own music library.  
This is a complete, production-ready setup with user authentication, background task queues, and serverless GPU processing.

---

## 🚀 Features

- 🎵 **AI Music Generation with ACE-Step**
- 🧠 **LLM-Powered Lyric & Prompt Generation** with Qwen2-7B
- 🖼️ **AI Thumbnail Generation** with stabilityai/sdxl-turbo
- 🎤 Multiple generation modes:
  - From a description
  - From custom lyrics
  - From described lyrics
- 🎸 **Instrumental Track Option** – generate music without vocals
- ⚡ **Serverless GPU Processing** with Modal for fast generation
- 📊 **Queue System** with Inngest for background task handling
- 👤 **User Authentication** with BetterAuth
- 🎧 **Community Music Feed** to discover, play, and like user-generated music
- 🎛️ **Personal Track Dashboard** to manage, play, and publish songs
- 🐍 **Python + FastAPI Backend** for generation logic
- 📱 **Modern UI** with Next.js, Tailwind CSS, and Shadcn UI

---

## 🛠 Tech Stack

**Frontend:**
- Next.js 15
- React 18
- TypeScript
- Tailwind CSS
- Shadcn UI
- BetterAuth (Authentication)

**Backend:**
- Python 3.12
- FastAPI
- [ACE-Step](https://pypi.org/project/ace-step/) (Music generation)
- Qwen2-7B (Lyric & prompt generation)
- Stability AI SDXL Turbo (Image generation)
- Inngest (Queue & background tasks)
- Modal (Serverless GPU execution)

**Infrastructure:**
- AWS S3 for storage
- Neon (PostgreSQL database)
- Vercel

---


