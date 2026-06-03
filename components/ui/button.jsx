import React from "react";

function mergeClasses(...classes) {
  return classes.filter(Boolean).join(" ");
}

export const Button = React.forwardRef(
  ({ className = "", variant = "default", type = "button", ...props }, ref) => {
    const variantClass =
      variant === "outline"
        ? "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50"
        : "bg-slate-900 text-white hover:bg-slate-800";

    return (
      <button
        ref={ref}
        type={type}
        className={mergeClasses(
          "inline-flex min-h-10 items-center justify-center rounded px-4 py-2 text-sm font-semibold transition disabled:pointer-events-none disabled:opacity-50",
          variantClass,
          className,
        )}
        {...props}
      />
    );
  },
);

Button.displayName = "Button";
