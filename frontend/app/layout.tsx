import type { Metadata } from "next";
import "./globals.css";
import ThemeToggle from "@/components/ThemeToggle";

export const metadata: Metadata = {
  title: "Sports AI",
  description: "Premier League prediction engine",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var stored = localStorage.getItem('theme');
                  var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                  var theme = stored || (prefersDark ? 'dark' : 'light');
                  if (theme === 'dark') {
                    document.documentElement.classList.add('dark');
                  } else {
                    document.documentElement.classList.remove('dark');
                  }
                } catch (e) {}
              })();
            `,
          }}
        />
      </head>
      <body className="antialiased bg-background text-foreground">
        <nav className="bg-card-bg border-b border-card-border px-8 py-3">
          <div className="max-w-4xl mx-auto flex gap-6 items-center">
            <a href="/" className="font-semibold hover:text-blue-500">Today</a>
            <a href="/predict" className="font-semibold hover:text-blue-500">Predict</a>
            <ThemeToggle />
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}