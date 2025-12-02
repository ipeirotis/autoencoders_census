import React, { useState, useRef } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export const Dropzone: React.FC<DropzoneProps> = ({
  onFileSelect,
  disabled = false,
}) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    if (disabled) return;

    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    if (disabled) return;

    setIsDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type === "text/csv" || file.name.endsWith(".csv")) {
        onFileSelect(file);
      } else {
        alert("Please drop a CSV file");
      }
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;

    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  return (
    <div
      data-testid="upload-dropzone"
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload CSV file - drop here or click to browse"
      aria-disabled={disabled}
      className={cn(
        "relative border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer",
        "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2",
        {
          "border-blue-500 bg-blue-50": isDragActive,
          "border-gray-300 hover:bg-gray-50": !isDragActive && !disabled,
          "border-gray-200 bg-gray-50 cursor-not-allowed opacity-60": disabled,
        }
      )}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,text/csv"
        onChange={handleInputChange}
        disabled={disabled}
        className="hidden"
        aria-hidden="true"
      />

      <div className="flex flex-col items-center gap-3">
        <Upload
          className={cn("w-8 h-8", {
            "text-blue-500": isDragActive,
            "text-gray-400": !isDragActive,
          })}
        />
        <p className="text-gray-800 font-medium">Drop CSV here or click to browse</p>
      </div>
    </div>
  );
};
