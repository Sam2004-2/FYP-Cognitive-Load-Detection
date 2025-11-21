/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'load-low': '#10b981',    // green-500
        'load-medium': '#f59e0b', // yellow-500  
        'load-high': '#ef4444',   // red-500
      },
    },
  },
  plugins: [],
}

