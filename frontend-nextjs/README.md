# DeepFake Detection Hub - Next.js Frontend

A modern, production-ready Next.js 15 frontend for the DeepFake Detection Hub application. Built with React 19, TypeScript, Tailwind CSS 4, and Framer Motion.

## Features

- 🎨 **Cyberpunk Aurora Theme** - Stunning dark theme with neon gradients and glassmorphism
- 🚀 **Next.js 15 App Router** - Latest React Server Components and streaming
- 📱 **Fully Responsive** - Mobile-first design that works on all devices
- ⚡ **Real-time Updates** - WebSocket integration for live analysis progress
- 🔒 **Type-Safe** - Full TypeScript support with strict mode
- 🎬 **Animated UI** - Smooth Framer Motion animations throughout
- 📊 **Data Visualization** - Interactive charts with Recharts

## Tech Stack

- **Framework**: Next.js 15.1+ (App Router)
- **Language**: TypeScript 5.9+
- **Styling**: Tailwind CSS 4.1+
- **UI Components**: Radix UI (shadcn/ui pattern)
- **Animations**: Framer Motion 12+
- **API Client**: Axios
- **WebSockets**: Socket.io-client
- **Charts**: Recharts

## Project Structure

```
frontend-nextjs/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Landing page
│   ├── analyze/            # Analysis page
│   ├── history/            # History & result pages
│   └── how-it-works/       # Info page
├── components/
│   ├── ui/                 # Reusable UI components
│   ├── features/           # Feature-specific components
│   └── layout/             # Header, Footer
├── hooks/                  # Custom React hooks
├── lib/
│   ├── api/                # API client & endpoints
│   ├── websocket/          # WebSocket utilities
│   └── utils/              # Helper functions
├── types/                  # TypeScript definitions
└── public/                 # Static assets
```

## Getting Started

### Prerequisites

- Node.js 18+ (recommended: 20+)
- npm or pnpm
- Running Flask backend on port 5000

### Installation

```bash
# Navigate to frontend directory
cd frontend-nextjs

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:3000`.

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:5000
NEXT_PUBLIC_WS_URL=http://localhost:5000
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm run type-check` | Run TypeScript check |

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with features & CTA |
| `/analyze` | File upload & analysis |
| `/history` | Previous analysis results |
| `/history/[id]` | Detailed result view |
| `/how-it-works` | Explanation of the detection process |

## API Integration

The frontend connects to the Flask backend via:

- **REST API** - For file uploads and data fetching
- **WebSocket** - For real-time analysis progress updates

## Building for Production

```bash
# Build the application
npm run build

# Start production server
npm run start
```

For Docker deployment, the app outputs a standalone build.

## License

MIT License
