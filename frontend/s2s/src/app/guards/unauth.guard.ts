import { CanDeactivateFn } from '@angular/router';

export const unauthGuard: CanDeactivateFn<unknown> = (component, currentRoute, currentState, nextState) => {
  return true;
};
