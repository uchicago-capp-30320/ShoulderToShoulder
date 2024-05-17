import { CanActivateFn } from '@angular/router';
import { ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { inject } from '@angular/core';

// services
import { AuthService } from '../_services/auth.service';

/**
 * Auth guard to check if user is logged in.
 *
 * @param route The route to activate
 * @param state The router state
 * @returns A boolean indicating if the user is logged in.
 */
export const canActivateProfile: CanActivateFn = (
  route: ActivatedRouteSnapshot,
  state: RouterStateSnapshot,
) => {
  return inject(AuthService).loggedIn;
};
