"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggle() {
    const root = document.documentElement;
    const nowDark = !root.classList.contains("dark");
    if (nowDark) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
    setIsDark(nowDark);
  }

  return (
    <button
      onClick={toggle}
      className="ml-auto text-sm px-3 py-1 rounded border border-card-border hover:bg-gray-100 dark:hover:bg-gray-800"
    >
      {isDark ? "☀️ Light" : "🌙 Dark"}
    </button>
  );
}