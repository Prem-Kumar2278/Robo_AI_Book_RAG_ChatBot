# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Development Setup

### Prerequisites
- Node.js 18+ installed
- Git installed
- Access to OpenAI API key (or alternative for RAG)
- Ubuntu 22.04 system with RTX GPU (for full compatibility testing)

### Initial Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd robo-ai-book
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create environment file:
   ```bash
   cp .env.example .env.local
   # Add your API keys to .env.local
   ```

4. Start the development server:
   ```bash
   npm start
   ```

5. Open `http://localhost:3000` in your browser

## Content Creation

### Adding a New Chapter
1. Navigate to the appropriate module directory in `docs/`
2. Create a new MDX file with the proper naming convention
3. Follow the standard chapter template with objectives, theory, code examples, exercises, and visuals

### Running Reproducibility Checks
```bash
npm run test:reproducibility
```

### Validating Accessibility
```bash
npm run test:accessibility
```

## Interactive Features

### RAG Chatbot
The RAG chatbot is integrated into all pages and can be accessed via the floating widget. To test:
1. Ensure the RAG backend is configured with your API keys
2. Ask questions about the current page content
3. Verify responses are relevant and accurate

### Personalization
User progress is tracked automatically when logged in. To enable personalization:
1. Register an account or log in
2. Navigate through content to build your progress profile
3. Check the dashboard for personalized recommendations

### Urdu Translation
Translation is available via the language selector in the header. To verify:
1. Click the language selector
2. Choose "Urdu"
3. Verify content is properly translated

## Building for Production

```bash
npm run build
```

## Deployment

The textbook is configured for deployment to GitHub Pages or Vercel:
- For GitHub Pages: `npm run deploy`
- For Vercel: Connect your repository to Vercel dashboard

## Troubleshooting

### Common Issues
- If pages don't load, check that the folder structure matches the sidebar configuration
- If RAG chatbot doesn't respond, verify API keys are properly configured
- For translation issues, check the i18n directory structure