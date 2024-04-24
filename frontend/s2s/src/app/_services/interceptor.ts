import { Injectable } from '@angular/core';
import { HttpInterceptor, HttpRequest, HttpHandler, HttpEvent } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ApiService } from './api.service';

@Injectable()
export class AuthInterceptor implements HttpInterceptor {
    constructor(
        private apiService: ApiService
    ) {}

    intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const authToken = this.apiService.appToken; // Assume a service that handles token retrieval
    const authReq = request.clone({
        headers: request.headers.set('Authorization', authToken)
    });
    return next.handle(authReq);
    }
}
