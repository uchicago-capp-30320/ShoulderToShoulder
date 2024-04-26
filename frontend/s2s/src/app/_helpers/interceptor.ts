import { Injectable } from '@angular/core';
import { HttpInterceptor, HttpRequest, HttpHandler, HttpEvent } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ApiService } from '../_services/api.service';

@Injectable()
export class AuthInterceptor implements HttpInterceptor {
    constructor(
        private apiService: ApiService
    ) {}

    intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
        let authToken = this.apiService.appToken; // Assume a service that handles token retrieval

        if (request.url.includes('onboarding') || request.url.includes('scenario')) {
            if (localStorage.getItem('access_token')) {
                authToken = `JWT ${localStorage.getItem('access_token') as string}`;
            }
        }

        const authReq = request.clone({
            headers: request.headers.set('Authorization', authToken),
        });
        
        return next.handle(authReq);
    }
}
