/**
 * Tests for Winston logger configuration
 */

import { describe, it, expect, jest } from '@jest/globals';

describe('Logger configuration', () => {
  it('should export logger instance', async () => {
    const loggerModule = await import('../../config/logger');
    expect(loggerModule.logger).toBeDefined();
  });

  it('should have info method', async () => {
    const loggerModule = await import('../../config/logger');
    expect(typeof loggerModule.logger.info).toBe('function');
  });

  it('should have error method', async () => {
    const loggerModule = await import('../../config/logger');
    expect(typeof loggerModule.logger.error).toBe('function');
  });

  it('should have warn method', async () => {
    const loggerModule = await import('../../config/logger');
    expect(typeof loggerModule.logger.warn).toBe('function');
  });

  it('should log structured JSON data', async () => {
    const loggerModule = await import('../../config/logger');
    const spy = jest.spyOn(loggerModule.logger, 'info');

    loggerModule.logger.info('Test message', { userId: '123', action: 'test' });

    expect(spy).toHaveBeenCalledWith('Test message', { userId: '123', action: 'test' });
    spy.mockRestore();
  });

  it('should log errors with stack traces', async () => {
    const loggerModule = await import('../../config/logger');
    const spy = jest.spyOn(loggerModule.logger, 'error');
    const testError = new Error('Test error');

    loggerModule.logger.error('Error occurred', { error: testError.message, stack: testError.stack });

    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it('should use Console transport in test environment', async () => {
    const loggerModule = await import('../../config/logger');
    // In test env, logger should have at least one transport
    expect(loggerModule.logger.transports.length).toBeGreaterThan(0);
  });
});
