/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        // Clean, professional, and trustworthy serifs for headings
        display: ['"Inter"', 'system-ui', 'sans-serif'],
        // High-legibility sans for clinical data
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        // Sharp mono for precise measurements/values
        mono: ['"Geist Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        bg: {
          base: '#F8FAFC',    // Soft "Hospital" white (less blue than pure white)
          elev1: '#FFFFFF',   // Pure white for cards
          elev2: '#F1F5F9',   // Light gray for secondary sections
          elev3: '#E2E8F0',   // Subtle borders/dividers
        },
        line: {
          DEFAULT: '#E2E8F0',
          strong: '#CBD5E1',
        },
        ink: {
          primary: '#0F172A',   // Deep navy instead of black for softer contrast
          secondary: '#475569', // Muted text for labels
          tertiary: '#94A3B8',  // Metadata/placeholder text
        },
        accent: {
          DEFAULT: '#0EA5E9',   // Trustworthy "Medical Blue"
          dim: '#0284C7',
          ghost: 'rgba(14, 165, 233, 0.08)',
        },
        // Refined clinical priority colors
        urgency: {
          p1: '#E11D48', // Emergent (Red)
          p2: '#F59E0B', // Urgent (Amber)
          p3: '#3B82F6', // Non-urgent (Blue)
          p4: '#10B981', // Routine (Green)
        },
        // Diagnostic categories (softer, professional tones)
        mech: {
          vasc: '#60A5FA', 
          degen: '#A855F7', 
          inflam: '#FB7185',
          dyst: '#FBBF24',
          struct: '#2DD4BF',
          tumor: '#F43F5E',
        },
      },
      boxShadow: {
        // Soft, organic shadows to feel modern and "comfortable"
        'clinical': '0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03)',
        'focus-ring': '0 0 0 3px rgba(14, 165, 233, 0.2)',
      },
      animation: {
        'fade-in': 'fadeIn 200ms ease-in-out',
        'slide-up': 'slideUp 300ms cubic-bezier(0.16, 1, 0.3, 1)',
        'pulse-soft': 'pulseSoft 3s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
        slideUp: { 
          '0%': { opacity: 0, transform: 'translateY(10px)' }, 
          '100%': { opacity: 1, transform: 'translateY(0)' } 
        },
        pulseSoft: { 
          '0%,100%': { opacity: 1 }, 
          '50%': { opacity: 0.7 } 
        },
      },
    },
  },
  plugins: [],
}