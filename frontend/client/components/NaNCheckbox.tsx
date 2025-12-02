import React from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

interface NaNCheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

export const NaNCheckbox: React.FC<NaNCheckboxProps> = ({
  checked,
  onChange,
  disabled = false,
}) => {
  return (
    <div className="flex items-center gap-3">
      <Checkbox
        data-testid="nan-checkbox"
        id="nan-as-category"
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
        className="w-5 h-5"
      />
      <Label
        htmlFor="nan-as-category"
        className="text-sm font-medium text-gray-700 cursor-pointer"
      >
        Treat NaN/missing values as its own category
      </Label>
    </div>
  );
};
