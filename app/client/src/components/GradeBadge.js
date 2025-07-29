// src/components/GradeBadge.jsx
import { motion } from 'framer-motion';

export default function GradeBadge({ label, color }) {
  return (
    <motion.div
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      className={`inline-block px-4 py-2 rounded-full text-2xl font-bold text-white shadow-lg 
                  ${color === 'bad'     ? 'bg-red-500' :
                    color === 'good'    ? 'bg-blue-400' :
                    color === 'great'   ? 'bg-yellow-400' :
                                         'bg-green-400'} 
                  animate-pulse`}
    >
      <span className="drop-shadow-lg">{label}</span>
    </motion.div>
  );
}