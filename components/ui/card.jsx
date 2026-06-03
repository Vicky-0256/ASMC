import React from "react";

function mergeClasses(...classes) {
  return classes.filter(Boolean).join(" ");
}

export function Card({ className = "", ...props }) {
  return <div className={mergeClasses("rounded-md bg-white", className)} {...props} />;
}

export function CardContent({ className = "", ...props }) {
  return <div className={className} {...props} />;
}
