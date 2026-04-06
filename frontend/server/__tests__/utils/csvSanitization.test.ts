import { sanitizeFormulaInjection } from '../../utils/csvSanitization';

describe('csvSanitization', () => {
  describe('sanitizeFormulaInjection', () => {
    it('should prefix formula starting with = with single quote', () => {
      expect(sanitizeFormulaInjection('=SUM(A1:A10)')).toBe("'=SUM(A1:A10)");
    });

    it('should prefix formula starting with + with single quote', () => {
      expect(sanitizeFormulaInjection('+5')).toBe("'+5");
    });

    it('should prefix formula starting with - with single quote', () => {
      expect(sanitizeFormulaInjection('-10')).toBe("'-10");
    });

    it('should prefix formula starting with @ with single quote', () => {
      expect(sanitizeFormulaInjection('@cmd')).toBe("'@cmd");
    });

    it('should prefix string starting with tab or carriage return with single quote', () => {
      expect(sanitizeFormulaInjection('\t\r\n')).toBe("'\t\r\n");
    });

    it('should not modify normal text', () => {
      expect(sanitizeFormulaInjection('normal text')).toBe('normal text');
    });

    it('should not modify non-string values', () => {
      expect(sanitizeFormulaInjection(42)).toBe(42);
    });

    it('should handle null and undefined gracefully', () => {
      expect(sanitizeFormulaInjection(null)).toBe(null);
      expect(sanitizeFormulaInjection(undefined)).toBe(undefined);
    });
  });
});
